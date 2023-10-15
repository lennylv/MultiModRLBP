import os
import sys

import pickle
import networkx as nx
import numpy as np
import random
import tarfile
import zipfile
import itertools
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import dgl
# from dgl.dataloading import EdgeDataLoader

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..', '..'))

from RGCN.kernels.node_sim import SimFunctionNode, k_block_list
from RGCN.utils import graph_io
from RGCN.data_loading.feature_maps import build_node_feature_parser
from RGCN.config.graph_keys import GRAPH_KEYS, TOOL, EDGE_MAP_RGLIB_REVERSE


class GraphDataset(Dataset):
    def __init__(self,
                 data_path=None,
                 hashing_path=None,
                 download_dir=None,
                 redundancy='NR',
                 chop=False,
                 annotated=False,
                 all_graphs=None,
                 edge_map=GRAPH_KEYS['edge_map'][TOOL],
                 label='LW',
                 node_simfunc=None,
                 node_features='nt_code',
                 node_target=None,
                 verbose=False):
        """
        This class is the main object for graph data loading. One can simply ask for feature and the appropriate data
        will be fetched.

        :param data_path: The path of the data. If node_sim is not None, this data should be annotated
        :param hashing_path: If node_sim is not None, we need hashing tables. If the path is not automatically created
        (ie the data was downloaded manually) one should input the path to the hashing.
        :param download_dir: When one fetches the data, one can choose where to dl it.
        By default, it will go to ~/.RGCN/
        :param redundancy: To use all graphs or just the non redundant set.
        :param chop: if we want full graphs or chopped ones for learning on smaller chunks
        :param annotated: if we want annotated graphs
        :param all_graphs: In the given directory, one can choose to provide a list of graphs to use
        :param edge_map: Necessary to build the one hot mapping from edge labels to an id
        :param label: The label to use
        :param node_simfunc: The node comparison object as defined in kernels/node_sim to use for the embeddings.
         If None is selected, this will just return graphs
        :param node_features: node features to include, stored in one tensor in order given by user,
        for example : ('nt_code','is_modified')
        :param node_features: node targets to include, stored in one tensor in order given by user
        for example : ('binding_protein', 'binding_small-molecule')
        :return:
        """

        # If we don't input a data path, the right one according to redundancy, chop and annotated is fetched
        # By default, we set hashing to None and potential node sim should be specified when creating
        # the node_sim function.
        # Then if a download occurs and no hashing was provided to the loader, the hashing used is the one
        # fetched by the downloading process to ensure it matches the data we iterate over.
        self.data_path = data_path
        self.hashing_path = hashing_path
        if data_path is None:
            self.data_path, self.hashing_path = graph_io.download_graphs(redundancy=redundancy,
                                                                         chop=chop,
                                                                         annotated=annotated,
                                                                         download_dir=download_dir,
                                                                         verbose=verbose)

        if all_graphs is not None:
            self.all_graphs = all_graphs
        else:
            self.all_graphs = sorted(os.listdir(self.data_path))

        # This is len() so we have to add the +1
        self.label = label
        self.edge_map = edge_map
        self.num_edge_types = max(self.edge_map.values()) + 1
        if verbose:
            print(f"Found {self.num_edge_types} relations")

        # If it is not None, add a node comparison tool
        self.node_simfunc, self.level = self.add_node_sim(node_simfunc=node_simfunc)

        # If queried, add node features and node targets
        self.node_features = [node_features] if isinstance(node_features, str) else node_features
        self.node_target = [node_target] if isinstance(node_target, str) else node_target

        self.node_features_parser = build_node_feature_parser(self.node_features)
        self.node_target_parser = build_node_feature_parser(self.node_target)

        self.input_dim = self.compute_dim(self.node_features_parser)
        self.output_dim = self.compute_dim(self.node_target_parser)
        self.norml = None

    def __len__(self):
        return len(self.all_graphs)
# node_features = [ 'nt_code', 'alpha', 'beta', 'gamma', 'delta', 
#                       'epsilon', 'zeta', 'epsilon_zeta',  'chi',  'C5prime_xyz',
#                         'P_xyz', 'ssZp', 'Dp', 'splay_angle', 'splay_distance', 'splay_ratio', 
#                         'eta', 'theta', 'eta_prime', 'theta_prime', 'eta_base', 'theta_base', 'v0', 'v1',
#                           'v2', 'v3', 'v4', 'amplitude', 'phase_angle',  
#                         'suiteness', 'filter_rmsd']
    def setNorm(self,norml):
        self.norml = norml
    def getNorm(self):
        if self.norml == None:
            return self.normalFeatures()
        else:
            return self.norml
    def normalFeatures(self):
        maxattr = dict()
        # for x in self.node_features:
        #         if x not in ['nt_code', 'alpha', 'beta', 'gamma', 'delta','epsilon', 'zeta', 'epsilon_zeta',  'chi', 'eta', 'theta', 'eta_prime','eta', 'theta', 'eta_prime','theta_prime', 'eta_base', 'theta_base','phase_angle']:
        #             maxattr[x] = [0]
        maxdeg = 0
        maxclose = 0
        for idx in range(len(self.all_graphs)):
            rna = self.all_graphs[idx][0:4]+'.json'
            g_path = os.path.join(self.data_path, rna)
            graph = graph_io.load_graph(g_path)
            
            
            graph = self.fix_buggy_edges(graph=graph)
            degrees = graph.degree()
            closeness = nx.closeness_centrality(graph)
            # print(degrees)
            deg_dic = dict()
            close_dic = dict()
            for node, degree in degrees:
                maxdeg = max(maxdeg,degree)
            for node, closeness_value in closeness.items():
                maxclose = max(maxclose, closeness_value)
            for x in self.node_features:
                if x not in ['nt_code', 'alpha', 'beta', 'gamma', 'splay_angle','delta','epsilon', 'zeta', 'epsilon_zeta',  'chi', 'eta', 'theta', 
                        'eta_prime','eta', 'theta', 'eta_prime','theta_prime', 'eta_base', 'theta_base','phase_angle']:
                    for node, attrs in graph.nodes.data():
                        # print(attrs[x])
                        if x not in maxattr.keys():
                            if isinstance(attrs[x],list):
                                maxattr[x] = [0]*len(attrs[x])
                            else:
                                maxattr[x] = [0]
                        if isinstance(attrs[x],list):
                            for i in range(len(attrs[x])):
                                if attrs[x][i] == None:
                                    break
                                if abs(attrs[x][i]) > maxattr[x][i]:
                                    maxattr[x][i] = abs(attrs[x][i])
                        else:
                            if attrs[x] == None:
                                break
                            if x in ['puckering','sugar_class','bin']:
                                maxattr[x][0] = 1
                                continue
                            if abs(attrs[x]) > maxattr[x][0]:
                                maxattr[x][0] = abs(attrs[x])
                        
        normalVector = []
        for x in self.node_features:
            if x in ['alpha', 'beta', 'gamma','splay_angle', 'delta','epsilon', 'zeta', 'epsilon_zeta',  'chi', 'eta', 'theta', 'eta_prime','eta', 'theta', 'eta_prime','theta_prime', 'eta_base', 'theta_base','phase_angle']:
                normalVector.append(180)
            elif x == 'nt_code':
                normalVector.extend([1,1,1,1])
            elif x == 'puckering':
                normalVector.extend([1,1,1,1,1,1,1,1,1,1,1])
            elif x == 'sugar_class':
                normalVector.extend([1,1,1])
            elif x == 'bin':
                normalVector.extend([1,1,1,1,1,1,1,1,1,1,1,1,1,1])
            else:
                normalVector.extend(maxattr[x])
        normalVector.extend([maxdeg,maxclose])
        self.norml = normalVector
        return normalVector
        pass
    def add_node_sim(self, node_simfunc):
        if node_simfunc is not None:
            if node_simfunc.method in ['R_graphlets', 'graphlet', 'R_ged']:
                if self.hashing_path is not None:
                    node_simfunc.add_hashtable(self.hashing_path)
                level = 'graphlet_annots'
            else:
                level = 'edge_annots'
        else:
            node_simfunc, level = None, None
        return node_simfunc, level

    def update_node_sim(self, node_simfunc):
        """
        This function is useful because the default_behavior is changed compared to above :
            Here if None is given, we don't remove the previous node_sim function

        :param node_simfunc: A nodesim.compare function
        :return:
        """
        if node_simfunc is not None:
            if node_simfunc.method in ['R_graphlets', 'graphlet', 'R_ged']:
                if self.hashing_path is not None:
                    node_simfunc.add_hashtable(self.hashing_path)
                level = 'graphlet_annots'
            else:
                level = 'edge_annots'
            self.node_simfunc, self.level = node_simfunc, level

    def get_node_encoding(self, g, encode_feature=True):
        """

        Get targets for graph g
        for every node get the attribute specified by self.node_target
        output a mapping of nodes to their targets

        :param g: a nx graph
        :param encode_feature: A boolean as to whether this should encode the features or targets
        :return: A dict that maps nodes to encodings
        """
        targets = {}
        node_parser = self.node_features_parser if encode_feature else self.node_target_parser

        if len(node_parser) == 0:
            return None
        degrees = g.degree()
        closeness = nx.closeness_centrality(g)
        # print(degrees)
        deg_dic = dict()
        close_dic = dict()
        for node, degree in degrees:
            deg_dic[node] = degree
        for node, closeness_value in closeness.items():
            close_dic[node]= closeness_value
        for node, attrs in g.nodes.data():
            all_node_feature_encoding = list()
            for i, (feature, feature_encoder) in enumerate(node_parser.items()):
                try:
                    node_feature = attrs[feature]
                    # print(node_feature )
                    node_feature_encoding = feature_encoder.encode(node_feature)
                except KeyError:
                    node_feature_encoding = feature_encoder.encode_default()
                all_node_feature_encoding.append(node_feature_encoding)
            if encode_feature:
                degAndClos = torch.tensor([deg_dic[node],close_dic[node]])
                all_node_feature_encoding.append(degAndClos)
            targets[node] = torch.cat(all_node_feature_encoding)
        return targets

    def compute_dim(self, node_parser):
        """
        Based on the encoding scheme, we can compute the shapes of the in and out tensors

        :return:
        """
        if len(node_parser) == 0:
            return 0
        all_node_feature_encoding = list()
        for i, (feature, feature_encoder) in enumerate(node_parser.items()):
            node_feature_encoding = feature_encoder.encode_default()
            all_node_feature_encoding.append(node_feature_encoding)
        all_node_feature_encoding = torch.cat(all_node_feature_encoding)
        return len(all_node_feature_encoding)

    def fix_buggy_edges(self, graph, strategy='remove'):
        """
        Sometimes some edges have weird names such as t.W representing a fuzziness.
        We just remove those as they don't deliver a good information

        :param graph:
        :param strategy: How to deal with it : for now just remove them.
        In the future maybe add an edge type in the edge map ?
        :return:
        """
        if strategy == 'remove':
            # Filter weird edges for now
            to_remove = list()
            for start_node, end_node, nodedata in graph.edges(data=True):
                if nodedata[self.label] not in self.edge_map:
                    to_remove.append((start_node, end_node))
            for start_node, end_node in to_remove:
                graph.remove_edge(start_node, end_node)
        else:
            raise ValueError(f'The edge fixing strategy : {strategy} was not implemented yet')
        return graph

    def shuffle(self):
        self.all_graphs = np.random.shuffle(self.all_graphs)

    def __getitem__(self, idx):

        rna = self.all_graphs[idx][0:4]+'.json'
        g_path = os.path.join(self.data_path, rna)
        graph = graph_io.load_graph(g_path)
        chain_id = self.all_graphs[idx].split('.')[0][5:]
        print(self.all_graphs[idx])
        #print(chain_id)

        graph = self.fix_buggy_edges(graph=graph)
        indxs = [ (item, data['nt_code'],data['index']) for item , data in graph.nodes(data=True)]
        # sorted_nodes = sorted(graph.nodes(), key=lambda n: (graph.nodes[n].split('.')[-2],graph.nodes[n].split('.')[-1]))
        # print(indxs)
        seq = sorted(indxs,key = lambda x: (x[0].split('.')[-2],int(x[0].split('.')[-1])))
        indlices = []#用于确认node对应序列的特征 是Bert输出的第几个
        for idx in range(len(seq)):
            indlices.append(indxs.index(seq[idx]))
        # print(indlices)
        chain = seq[0][0].split('.')[-2]
        subseq = []
        seqs = []
        chain_id_list = [chain,]
        chain_idx = -1
        for id, nt, idx in seq:
            if(chain != id.split('.')[-2]):
                chain = id.split('.')[-2]
                seqs.append(subseq)
                chain_id_list.append(chain)
                subseq = []
            subseq.append(nt.upper())
        seqs.append(subseq)
        #print(chain_id_list)
        for x in range(len(chain_id_list)):
            # print(x)
            if(chain_id_list[x].lower() == chain_id):
                chain_idx = x
                break
        if chain_idx == -1: chain_idx = 0
        if chain_id !='':
            chain_id_list=[chain_idx]
        else:
            chain_id_list = [ x  for x in range(len(chain_id_list))]
        #print(chain_id_list)
        # print(seq)
        # seqs = graph.graph['dbn']['all_chains']['bseq'].upper()
        # print(seqs)
        # print(graph.nodes(data=True))
        # print(indxs)
        # Get Edge Labels
        edge_type = {edge: torch.tensor(self.edge_map[label]) for edge, label in
                     (nx.get_edge_attributes(graph, self.label)).items()}
        nx.set_edge_attributes(graph, name='edge_type', values=edge_type)

        # Get Node labels
        node_attrs_toadd = list()
        if len(self.node_features_parser) > 0:
            feature_encoding = self.get_node_encoding(graph, encode_feature=True)
            
            # for key, item in feature_encoding:
            #         pass
            # print(feature_encoding)
            nx.set_node_attributes(graph, name='features', values=feature_encoding)
            node_attrs_toadd.append('features')
        if len(self.node_target_parser) > 0:
            target_encoding = self.get_node_encoding(graph, encode_feature=False)
            nx.set_node_attributes(graph, name='target', values=target_encoding)
            node_attrs_toadd.append('target')
        # Careful ! When doing this, the graph nodes get sorted.
        g_dgl = dgl.from_networkx(nx_graph=graph,
                                  edge_attrs=['edge_type'],
                                  node_attrs=node_attrs_toadd)
        # print(g_dgl.ndata['features'][:,0:4])
        if self.node_simfunc is not None:
            ring = list(sorted(graph.nodes(data=self.level)))
            
            return g_dgl, ring
        else:
            # print(g_path)
            # print(indxs)
            seqlength = [len(seq) for seq in seqs]
            return g_dgl, indlices, seqs, seqlength, chain_id_list


class UnsupervisedDataset(GraphDataset):
    def __init__(self,
                 node_simfunc=SimFunctionNode('R_1', 2),
                 annotated=True,
                 chop=True,
                 **kwargs):
        """
        Basically just change the default of the loader based on the usecase
        """
        super().__init__(annotated=annotated, chop=chop, node_simfunc=node_simfunc, **kwargs)


class SupervisedDataset(GraphDataset):
    def __init__(self,
                 node_target='binding_protein',
                 annotated=False,
                 **kwargs):
        """
        Basically just change the default of the loader based on the usecase
        """
        super().__init__(annotated=annotated, node_target=node_target, **kwargs)

def make_dict(k=3):
    # seq to num 
    l = ["A", "U", "G", "C"]
    kmer_list = [''.join(v) for v in list(itertools.product(l, repeat=k))]
    kmer_list.insert(0, "MASK")
    dic = {kmer: i+1 for i,kmer in enumerate(kmer_list)}
    return dic
def convert(seqs, kmer_dict, max_length):
    # 文字 数字
    seq_num = np.array([],type(torch.int32))
    if not max_length:
        max_length = max([len(i) for i in seqs])
    all_num =  np.array([],type(torch.int32)).reshape(-1,max_length)
    for s in seqs:
        for s2 in s:
            convered_seq = []
            for i in s2:
                if i in kmer_dict.keys():
                    convered_seq.append(kmer_dict[i])
                else:
                    convered_seq.append(1)   
            convered_seq =convered_seq + [0]*(max_length - len(s2))
            seq_num = np.append(seq_num, np.array(convered_seq), axis=0)
        seq_num = seq_num.reshape(-1, max_length)
        all_num = np.append(all_num, seq_num, axis=0)
        seq_num = np.array([],type(torch.int32))
    return all_num
def collate_wrapper(node_simfunc=None, max_size_kernel=None,normal = None):
    """
    Wrapper for collate function so we can use different node similarities.
        We cannot use functools.partial as it is not picklable so incompatible with Pytorch loading

    :param node_simfunc: A node comparison function as defined in kernels, to optionally return a pairwise comparison
    of the nodes in the batch
    :param max_size_kernel: If the node comparison is not None, optionnaly only return a pairwise comparison between
    a subset of all nodes, of size max_size_kernel
    :return: a picklable python function that can be called on a batch by Pytorch loaders
    """
    if node_simfunc is not None:
        def collate_block(samples):
            # The input `samples` is a list of tuples (graph, ring).
            # print("collate 1")
            graphs, rings = map(list, zip(*samples))
            # print(graphs)
            # print(rings)
            # DGL makes batching by making all small graphs a big one with disconnected components
            # We keep track of those
            batched_graph = dgl.batch(graphs)
            len_graphs = [graph.number_of_nodes() for graph in graphs]
            # print("collate 2")
            # Now compute similarities, we need to flatten the list and then use the kernels :
            # The rings is now a list of list of tuples
            # If we have a huge graph, we can sample max_size_kernel nodes to avoid huge computations,
            # We then return the sampled ids
            flat_rings = list()
            for ring in rings:
                flat_rings.extend(ring)
            # print("collate 3")
            if max_size_kernel is None or len(flat_rings) < max_size_kernel:
                # Just take them all
                node_ids = [1 for _ in flat_rings]
            else:
                # Take only 'max_size_kernel' elements
                node_ids = [1 for _ in range(max_size_kernel)] + \
                           [0 for _ in range(len(flat_rings) - max_size_kernel)]
                random.shuffle(node_ids)
                flat_rings = [node for i, node in enumerate(flat_rings) if node_ids[i] == 1]
            # print("collate 4")
            K = k_block_list(flat_rings, node_simfunc)
            return batched_graph, torch.from_numpy(K).detach().float(), len_graphs, node_ids
    else:
        def collate_block(samples):
            # The input `samples` is a list of pairs
            #  (graph, label).
            graphs, indxs, seqs, lens, chain_idx = map(list, zip(*samples))
            #print(chain_idx)
            seqlens = []
            for seqlen in lens:
                 seqlens.extend(seqlen)
            pre_num = 0
            for idx in range(len(chain_idx)):
                 for i in range(len(chain_idx[idx])):
                      chain_idx[idx][i] += pre_num
                 pre_num += len(lens[idx])
            chain_list = []
            for idx in range(len(chain_idx)):
                 chain_list.extend(chain_idx[idx])
            #for idx in range(len(lens)):
                 #chain_idx[idx] += pre_num
                 #pre_num += len(lens[idx])
            #print(chain_list)
            length = 0
            combindx = []
            for idx in indxs:
                for i in idx:
                    combindx.append(i+length)
                length = length + len(idx)
            k = 1
            kmer_dict = make_dict(k)
            
            seqs = convert(seqs, kmer_dict, 440)
            len_graphs = [graph.number_of_nodes() for graph in graphs]
            node_features = torch.tensor([])
            #print(seqs)
            # print(graphs)
            for graph in graphs:
                node_features = torch.cat((node_features,graph.ndata['features']),0)
                # print(node_features.shape)
                # print(graph.ndata['features'][0:4])
                # for i in range(graph.number_of_nodes()):
                # print(graph.ndata['features'].shape)
                if normal !=None:
                    node_features = torch.div(node_features,normal)
                    # for x in node_features:
                        # print(x)
                # only use nt_code to produce embeding
                graph.ndata['features'] = graph.ndata['features'][:,0:4]

            # print(graphs)
            batched_graph = dgl.batch(graphs)
            # node_features = [ graph.ndata['features']  for graph in graphs]
            # print(node_features[0])
           
            return batched_graph, combindx, node_features, seqs, seqlens, len_graphs, chain_list
    return collate_block


class GraphLoader:
    def __init__(self,
                 dataset,
                 batch_size=5,
                 num_workers=0,
                 max_size_kernel=None,
                 split=True,
                 verbose=False):
        """
        Turns a dataset into a dataloader

        :param dataset: The dataset to iterate over
        :param batch_size: The desired batch size (number of whole graphs)
        :param num_workers: The number of cores to use for loading. Defaults to 0 to match the PyTorch default.
        :param max_size_kernel: If we use K comptutations, we need to subsamble some nodes for the big graphs
        or else the k computation takes too long
        :param split: To return subsets to split the data
        :param verbose: To print some info about the data
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_size_kernel = max_size_kernel
        self.split = split
        self.verbose = verbose

    def get_data(self):
        print("start get data")
        normal = self.dataset.getNorm()
        print(normal)
        normal = torch.tensor(normal)
        collate_block = collate_wrapper(self.dataset.node_simfunc, max_size_kernel=self.max_size_kernel, normal=normal)
        if not self.split:
            loader = DataLoader(dataset=self.dataset, shuffle=True, batch_size=self.batch_size,
                                num_workers=self.num_workers, collate_fn=collate_block)
            return loader

        else:
            n = len(self.dataset)
            indices = list(range(n))
            np.random.seed(0)
            split_train, split_valid = 0.7, 0.85
            train_index, valid_index = int(split_train * n), int(split_valid * n)

            train_indices = indices[:train_index]
            valid_indices = indices[train_index:valid_index]
            test_indices = indices[valid_index:]

            train_set = Subset(self.dataset, train_indices)
            valid_set = Subset(self.dataset, valid_indices)
            test_set = Subset(self.dataset, test_indices)

            if self.verbose:
                print(f"training items: ", len(train_set))
            train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=self.batch_size,
                                      num_workers=self.num_workers, collate_fn=collate_block)
            valid_loader = DataLoader(dataset=valid_set, shuffle=True, batch_size=self.batch_size,
                                      num_workers=self.num_workers, collate_fn=collate_block)
            test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=self.batch_size,
                                     num_workers=self.num_workers, collate_fn=collate_block)
            return train_loader, valid_loader, test_loader


class InferenceLoader:
    def __init__(self,
                 list_to_predict,
                 data_path,
                 dataset=None,
                 batch_size=5,
                 num_workers=20,
                 **kwargs):
        if dataset is None:
            dataset = GraphDataset(data_path=data_path, **kwargs)
        self.dataset = dataset
        self.dataset.all_graphs = list_to_predict
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_data(self):
        collate_block = collate_wrapper(None)
        train_loader = DataLoader(dataset=self.dataset,
                                  shuffle=False,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  collate_fn=collate_block)
        return train_loader


class EdgeLoaderGenerator:
    def __init__(self,
                 graph_loader,
                 inner_batch_size=50,
                 sampler_layers=2,
                 neg_samples=1):
        """
        This turns a graph dataloader or dataset into an edge data loader generator.
        It needs to be reinitialized every epochs because of the double iteration pattern

        Iterates over batches of base pairs and generates negative samples for each.
        Negative sampling is just uniform for the moment (eventually we should change it to only sample
        edges at a certain backbone distance.

        timing :
        - num workers should be used to load the graphs not in the inner loop
        - The inner batch size yields huge speedups (probably generating all MFGs is tedious)

        :param graph_loader: A GraphLoader or GraphDataset. We will iterate over its graphs and then over its basepairs
        :param inner_batch_size: The amount of base-pairs to sample in each batch on each graph
        :param sampler_layers: The size of the neighborhood
        :param neg_samples: The number of negative sample to use per positive ones
        :param num_workers: The amount of cores to use for loading
        """
        self.graph_loader = graph_loader
        self.neg_samples = neg_samples
        self.sampler_layers = sampler_layers
        self.inner_batch_size = inner_batch_size
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.sampler_layers)
        self.negative_sampler = dgl.dataloading.negative_sampler.Uniform(self.neg_samples)
        self.eloader_args = {
            'shuffle': False,
            'batch_size': self.inner_batch_size,
            'negative_sampler': self.negative_sampler
        }

    @staticmethod
    def get_base_pairs(g):
        """
        Get edge IDS of edges in a base pair (non-backbone or unpaired).

        :param g: networkx graph
        :return: list of ids
        """
        eids = []
        for ind, e in enumerate(g.edata['edge_type']):
            if EDGE_MAP_RGLIB_REVERSE[e.item()][0] != 'B':
                eids.append(e)
        return eids

    def get_edge_loader(self):
        """
        Simply get the loader for one epoch. This needs to be called at each epoch

        :return: the edge loader
        """
        edge_loader = (EdgeDataLoader(g_batched, self.get_base_pairs(g_batched), self.sampler, **self.eloader_args)
                       for g_batched, _ in self.graph_loader)
        return edge_loader


class DefaultBasePairLoader:
    def __init__(self,
                 dataset=None,
                 data_path=None,
                 batch_size=5,
                 inner_batch_size=50,
                 sampler_layers=2,
                 neg_samples=1,
                 num_workers=4,
                 **kwargs):
        """
        Just a default edge base pair loader that deals with the splits

        :param dataset: A GraphDataset we want to loop over for base-pair prediction
        :param data_path: Optionnaly, we can use a data path to create a default GraphDataset
        :param batch_size: The desired batch size (number of whole graphs)
        :param inner_batch_size:The desired inner batch size (number of sampled edge in a batched graph)
        :param sampler_layers: The size of the neighborhood
        :param neg_samples: The number of negative sample to use per positive ones
        :param num_workers: The number of cores to use for loading
        """
        # Create default loaders
        if dataset is None:
            dataset = GraphDataset(data_path=data_path, **kwargs)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.g_train, self.g_val, self.g_test = GraphLoader(self.dataset,
                                                            batch_size=self.batch_size,
                                                            num_workers=self.num_workers).get_data()

        # Get the inner loader parameters
        self.inner_batch_size = inner_batch_size
        self.neg_samples = neg_samples
        self.sampler_layers = sampler_layers

    def get_data(self):
        train_loader = EdgeLoaderGenerator(graph_loader=self.g_train, inner_batch_size=self.inner_batch_size,
                                           sampler_layers=self.sampler_layers,
                                           neg_samples=self.neg_samples).get_edge_loader()
        val_loader = EdgeLoaderGenerator(graph_loader=self.g_val, inner_batch_size=self.inner_batch_size,
                                         sampler_layers=self.sampler_layers,
                                         neg_samples=self.neg_samples).get_edge_loader()
        test_loader = EdgeLoaderGenerator(graph_loader=self.g_test, inner_batch_size=self.inner_batch_size,
                                          sampler_layers=self.sampler_layers,
                                          neg_samples=self.neg_samples).get_edge_loader()

        return train_loader, val_loader, test_loader


if __name__ == '__main__':
    pass
    import time
    from RGCN.kernels import node_sim
    node_features = ['nt_code']
    node_featuress = [ 'nt_code', 'alpha', 'beta', 'gamma', 'delta', 
                      'epsilon', 'zeta', 'epsilon_zeta',  'chi',  'C5prime_xyz',
                        'P_xyz', 'form', 'ssZp', 'Dp', 'splay_angle', 'splay_distance', 'splay_ratio', 
                        'eta', 'theta', 'eta_prime', 'theta_prime', 'eta_base', 'theta_base', 'v0', 'v1',
                          'v2', 'v3', 'v4', 'amplitude', 'phase_angle', 'puckering',   
                           'suiteness', 'filter_rmsd', 'frame_rmsd', 'frame_origin', 'frame_x_axis', 
                          'frame_y_axis', 'frame_z_axis', 'frame_quaternion']
    node_target = ['binding_ion']
    # node_sim_func = node_sim.SimFunctionNode(method='R_graphlets', depth=2)
    node_sim_func = None
    # GET THE DATA GOING
    f = open('test/train60.txt','r')
    train60 = f.readlines()
    train60 = [x.split('\n')[0]+'.json' for x in train60]
    print(train60)
    toy_dataset = GraphDataset(data_path='data\\myData', 
                               hashing_path = "data\\2.5DGraph\iguana\\all_graphs_annot_hash.p",
                               node_simfunc= node_sim_func,
                               node_features=node_features,
                               node_target=node_target,
                               all_graphs=train60)
    # print(toy_dataset[1])
    train_loader, validation_loader, test_loader = GraphLoader(dataset=toy_dataset,
                                                               batch_size=5,
                                                               num_workers=0).get_data()
    # print(len(train_loader))
    start = time.time()
    for i, item in enumerate(toy_dataset):
        # print(i)
        # pickle.dump(item,
        #             open('D:\wjk\database\MyRNAPredict\data\\2.5DGraph\iguana\\all_graphs_annot_pre_ntcode/'+i),'wb')
        # print(item)
        if(i%20 == 0):
            end = time.time()
            print(f"{i}  {(end-start)/60}")
            start = time.time()


        # if i > 50:
        #     break
        # if not i % 20: print(i)
        pass
