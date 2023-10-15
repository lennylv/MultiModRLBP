import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from Rnaglib.rnattentional.layers import RGATLayer


class RGATEmbedder(nn.Module):
    """
    This is an exemple RGCN for unsupervised learning, going from one element of "dims" to the other

    It maps the "features" of an input graph to an "h" node attribute and returns the corresponding tensor.
    """

    def __init__(self,
                 dims,
                 num_heads=3,
                 sample_other=0.2,
                 infeatures_dim=0,
                 num_rels=20,
                 num_bases=None,
                 conv_output=True,
                 self_loop=True,
                 return_loss=True,
                 verbose=False):
        super(RGATEmbedder, self).__init__()
        self.dims = dims
        self.num_heads = num_heads
        self.sample_other = sample_other
        self.use_node_features = (infeatures_dim != 0)
        self.in_dim = 1 if infeatures_dim == 0 else infeatures_dim
        self.conv_output = conv_output
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.self_loop = self_loop
        self.verbose = verbose
        self.return_loss = return_loss
        
        self.layers = self.build_model()

        if self.verbose:
            print(self.layers)
            print("Num rels: ", self.num_rels)

    def build_model(self):
        layers = nn.ModuleList()

        short = self.dims[:-1]
        last_hidden, last = self.dims[-2:]
        if self.verbose:
            print("short, ", short)
            print("last_hidden, last ", last_hidden, last)

        # input feature is just node degree
        i2h = RGATLayer(in_feat=self.in_dim,
                        out_feat=self.dims[0],
                        num_rels=self.num_rels,
                        num_bases=self.num_bases,
                        num_heads=self.num_heads,
                        sample_other=self.sample_other,
                        activation=F.relu,
                        self_loop=self.self_loop)
        layers.append(i2h)

        for dim_in, dim_out in zip(short, short[1:]):
            h2h = RGATLayer(in_feat=dim_in * self.num_heads,
                            out_feat=dim_out,
                            num_rels=self.num_rels,
                            num_bases=self.num_bases,
                            num_heads=self.num_heads,
                            sample_other=self.sample_other,
                            activation=F.relu,
                            self_loop=self.self_loop)
            layers.append(h2h)

        # hidden to output
        if self.conv_output:
            h2o = RGATLayer(in_feat=last_hidden * self.num_heads,
                            out_feat=last,
                            num_rels=self.num_rels,
                            num_bases=self.num_bases,
                            num_heads=self.num_heads,
                            sample_other=self.sample_other,
                            self_loop=self.self_loop,
                            activation=None)
        else:
            h2o = nn.Linear(last_hidden * self.num_heads, last)
        layers.append(h2o)
        return layers

    def deactivate_loss(self):
        for layer in self.layers:
            if isinstance(layer, RGATLayer):
                layer.deactivate_loss()

    @property
    def current_device(self):
        """
        :return: current device this model is on
        """
        return next(self.parameters()).device

    def forward(self, g, features,mod = 1):
        iso_loss = 0
        if self.use_node_features:
            if mod == 1:
                h = g.ndata['features'].to(self.current_device)
            else:
                h = features
        else:
            # h = g.in_degrees().view(-1, 1).float().to(self.current_device)
            h = torch.ones(len(g.nodes())).view(-1, 1).to(self.current_device)
        for i, layer in enumerate(self.layers):
            if not self.conv_output and (i == len(self.layers) - 1):
                h = layer(h)
            else:
                if layer.return_loss:
                    h, loss = layer(g=g, feat=h)
                    iso_loss += loss
                else:
                    #print(features.shape)
                    h = layer(g=g, feat=h)
                    #print(h.shape)
        if self.return_loss:
            return h, iso_loss
        else:
            return h

import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads

        # 定义查询、键、值的线性变换层
        self.q_linear = nn.Linear(self.head_dim, out_features)
        self.k_linear = nn.Linear(self.head_dim, out_features)
        self.v_linear = nn.Linear(self.head_dim, out_features)
        #维度变换
        self.proj_linear = nn.Linear(in_features, self.head_dim * self.num_heads)
    def forward(self, x):
        # 将输入矩阵 x 分割成 num_heads 份
        seq_len = x.shape[0]
        # batch_size = x.shape[0]
        x = x.unsqueeze(0)
        batch_size = 1
        #x = self.proj_linear(x)  # (batch_size, seq_len, self.num_heads * self.head_dim)
        x = x.view(batch_size,seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 对查询、键、值进行线性变换
        q = self.q_linear(x)  # (batch_size, num_heads, seq_len, head_dim)
        k = self.k_linear(x)  # (batch_size, num_heads, seq_len, head_dim)
        v = self.v_linear(x)  # (batch_size, num_heads, seq_len, head_dim)
        # 计算注意力得分
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = F.softmax(scores, dim=-1)

        # 对节点特征进行加权求和
        out = torch.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(seq_len, -1)  # (batch_size, seq_len, num_heads * head_dim)
        return out.squeeze(0)
import torch
import torch.nn as nn

class MLNet(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob=0.5):
        super(MLNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_prob)
        # self.fc3 = nn.Linear(hidden_size, hidden_size)
        # self.bn3 = nn.BatchNorm1d(hidden_size)
        # self.dropout3 = nn.Dropout(dropout_prob)
        self.out = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.bn1(torch.relu(self.fc1(x))))
        x = self.dropout2(self.bn2(torch.relu(self.fc2(x))))
        # x = self.dropout3(self.bn3(torch.relu(self.fc3(x))))
        x = self.sigmoid(self.out(x))
        return x

class RGATClassifier(nn.Module):
    """
    This is an exemple RGCN for supervised learning, that uses the previous Embedder network

    It maps the "features" of an input graph to an "h" node attribute and returns the corresponding tensor.
    """

    def __init__(self,
                 rgat_embedder,
                 rbert_embedder = None,
                 rgat_embedder_pre = None,
                 classif_dims=None,
                 num_heads=5,
                 num_rels=20,
                 num_bases=None,
                 conv_output=True,
                 self_loop=True,
                 verbose=False,
                 return_loss=True,
                 sample_other=0.2):
        super(RGATClassifier, self).__init__()
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.self_loop = self_loop
        self.conv_output = conv_output
        self.num_heads = num_heads
        self.sample_other = sample_other
        self.return_loss = return_loss
        if rbert_embedder != None:
            self.bert_dim = 120
        else:
            self.bert_dim = 0
        self.feature_dim = 71
        self.rgat_embedder = rgat_embedder
        self.rgat_embedder_pre = rgat_embedder_pre
        self.last_dim_embedder = rgat_embedder.dims[-1] * rgat_embedder.num_heads + self.feature_dim + self.bert_dim + self.feature_dim + 128 * 1
        # noRGCN
        # self.last_dim_embedder =  self.feature_dim + self.bert_dim + self.feature_dim + 128 * 1
        #noBert

        # self.last_dim_embedder = rgat_embedder.dims[-1] * rgat_embedder.num_heads + self.feature_dim  + self.feature_dim + 128 * 1
        #noMid
        # self.last_dim_embedder = rgat_embedder.dims[-1] * rgat_embedder.num_heads + self.bert_dim 
        if self.rgat_embedder_pre!=None:
            self.last_dim_embedder += rgat_embedder_pre.dims[-1] * rgat_embedder_pre.num_heads
        self.classif_dims = classif_dims
        self.rbert_embedder = rbert_embedder	
        self.classif_layers = self.build_model()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.verbose = verbose
        if self.verbose:
            print(self.classif_layers)
            print("Num rels: ", self.num_rels)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.5)
        
        self.window = 11
        self.padsize = int(self.window/2)
        self.localpool = nn.MaxPool2d(kernel_size= 11, stride=1)
        self.localpad = nn.ConstantPad2d((self.padsize, self.padsize,self.padsize,self.padsize), value=0)
        


        #att
        att_input =350
        att_output =128
        att_num_head =8
        att_mlpinput = att_output * att_num_head
        att_mlphidden = 32
        self.att = Attention(att_input, att_output, att_num_head)
        self.mlp = MLNet(att_mlpinput, att_mlphidden)
        self.cutoff_len = 440

        kernels = [13, 17, 21]   
        padding1 = (kernels[1]-1)//2
        self.conv2d_1 = torch.nn.Sequential()
        self.conv2d_1.add_module("conv2d_1",torch.nn.Conv2d(1,128,padding= (padding1,0),kernel_size=(kernels[1],self.feature_dim)))
        #self.conv2d.add_module("relu1",nn.BatchNorm2d(128))
        self.conv2d_1.add_module("relu1",torch.nn.ReLU())
        self.conv2d_1.add_module("pool2",torch.nn.MaxPool2d(kernel_size= (self.cutoff_len,1),stride=1))
        
        # padding2 = (kernels[1]-1)//2
        # self.conv2d_2 = torch.nn.Sequential()
        # self.conv2d_2.add_module("conv2d_2",torch.nn.Conv2d(1,128,padding= (padding2,0),kernel_size=(kernels[1],self.feature_dim)))
        # #self.conv2d.add_module("relu1",nn.BatchNorm2d(128))
        # self.conv2d_2.add_module("relu1_2",torch.nn.ReLU())
        # # self.conv2d_2.add_module("pool2_2",torch.nn.MaxPool2d(kernel_size= (self.cutoff_len,1),stride=1))
        # self.conv2d_2_pool = torch.nn.MaxPool2d(kernel_size= (self.cutoff_len,1),stride=1)

        # padding3 = (kernels[2]-1)//2
        # self.conv2d_3 = torch.nn.Sequential()
        # self.conv2d_3.add_module("conv2d_3",torch.nn.Conv2d(1,128,padding= (padding3,0),kernel_size=(kernels[2],self.feature_dim)))
        # #self.conv2d.add_module("relu1",nn.BatchNorm2d(128))
        # self.conv2d_3.add_module("relu1_3",torch.nn.ReLU())
        # self.conv2d_3.add_module("pool2_3",torch.nn.MaxPool2d(kernel_size= (self.cutoff_len,1),stride=1))

        self.DNN1 = nn.Sequential()
        self.DNN1.add_module("Dense1", torch.nn.Linear(self.last_dim_embedder,192))
        self.DNN1.add_module("Relu1", torch.nn.ReLU())
        self.dropout_layer = nn.Dropout(0.1)
        self.DNN2 = nn.Sequential()
        self.DNN2.add_module("Dense2", torch.nn.Linear(192,96))
        self.DNN2.add_module("Relu2", torch.nn.ReLU())
        self.dropout_layer2 = nn.Dropout(0.1)
        self.outLayer = nn.Sequential(
            torch.nn.Linear(96, 1),
            torch.nn.Sigmoid())
        
        self.fc2d1 = nn.Linear(128, 64)
        self.fc2d2 = nn.Linear(64, 1)

        
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        for kernel_size in [5, 10, 20]:
            padding = (kernel_size - 1) // 2
            conv = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=kernel_size, padding=padding)
            bn = nn.BatchNorm1d(32)
            pool = nn.MaxPool1d(kernel_size=2)
            self.conv_layers.append(conv)
            self.bn_layers.append(bn)
            self.pool_layers.append(pool)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv11 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=100)
        self.conv12 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=175)
        self.conv13 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=350)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 172, 128)#84 191  64 84
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 1)
       
    def build_model(self):
        if self.classif_dims is None:
            return self.rgat_embedder

        classif_layers = nn.ModuleList()
        # Just one convolution
        if len(self.classif_dims) == 1:
            if self.conv_output:
                h2o = RGATLayer(in_feat=self.last_dim_embedder,
                                out_feat=self.classif_dims[0],
                                num_rels=self.num_rels,
                                num_bases=self.num_bases,
                                num_heads=self.num_heads,
                                sample_other=self.sample_other,
                                self_loop=self.self_loop,
                                # Old fix for a bug in dgl<0.6
                                # self_loop=self.self_loop and self.classif_dims[0] > 1,
                                activation=None)
            else:
                h2h = nn.Linear(self.last_dim_embedder, self.last_dim_embedder)
                classif_layers.append(h2h)
                h2h2 = nn.Linear(self.last_dim_embedder, self.last_dim_embedder)
                classif_layers.append(h2h2)
                h2o = nn.Linear(self.last_dim_embedder, self.classif_dims[0])
                 
            classif_layers.append(h2o)
            
            return classif_layers

        # The supervised is more than one layer
        else:
            i2h = RGATLayer(in_feat=self.last_dim_embedder,
                            out_feat=self.classif_dims[0],
                            num_rels=self.num_rels,
                            num_bases=self.num_bases,
                            num_heads=self.num_heads,
                            sample_other=self.sample_other,
                            activation=F.relu,
                            self_loop=self.self_loop)
            classif_layers.append(i2h)
            last_hidden, last = self.classif_dims[-2:]
            short = self.classif_dims[:-1]
            for dim_in, dim_out in zip(short, short[1:]):
                h2h = RGATLayer(in_feat=dim_in * self.num_heads,
                                out_feat=dim_out,
                                num_rels=self.num_rels,
                                num_bases=self.num_bases,
                                num_heads=self.num_heads,
                                sample_other=self.sample_other,
                                activation=F.relu,
                                self_loop=self.self_loop)
                classif_layers.append(h2h)

            # hidden to output
            if self.conv_output:
                h2o = RGATLayer(in_feat=last_hidden * self.num_heads,
                                out_feat=last,
                                num_rels=self.num_rels,
                                num_bases=self.num_bases,
                                num_heads=self.num_heads,
                                sample_other=self.sample_other,
                                self_loop=self.self_loop,
                                activation=None)
            else:
                h2o = nn.Linear(last_hidden * self.num_heads, last)
            classif_layers.append(h2o)
            return classif_layers

    def deactivate_loss(self):
        self.return_loss = False
        self.rgat_embedder.deactivate_loss()
        for layer in self.classif_layers:
            if isinstance(layer, RGATLayer):
                layer.deactivate_loss()

    @property
    def current_device(self):
        """
        :return: current device this model is on
        """
        return next(self.parameters()).device

    def forward(self, g , features, indxs, seqs, seqlens, len_graphs, chain_idx):
        iso_loss = 0
        g_copy = g.clone().to(self.current_device)
        cnt = 1

        if self.rgat_embedder.return_loss:
            h, loss = self.rgat_embedder(g,features,0)
        else:
            h = self.rgat_embedder(g,features,0)
            loss = 0
        if self.rgat_embedder_pre!=None:
            cnt = 2
            if self.rgat_embedder_pre.return_loss:
                #print(next(self.rgat_embedder_pre.parameters()).device)
            
                h_pre, loss_pre = self.rgat_embedder_pre(g_copy,features,1)
                h = torch.cat((h,h_pre),1)

            else:
                h_pre = self.rgat_embedder_pre(g_copy,features,1)
                h = torch.cat((h,h_pre),1)
                loss = 0
        iso_loss += loss
        # print(features)
        h = torch.cat((h,features),1)
        features = torch.index_select(features, 0, indxs)
        seq_embedding = torch.tensor([]).to(self.current_device).reshape(-1,120)

        h = torch.index_select(h, 0, indxs)

        # noRGCN
        # h = features
        #noMid
        noMid = 1
        # chain_len = torch.tensor([seqlens[x] for x in chain_idx])
        prediction_scores, prediction_scores_ss, seq_encode = self.rbert_embedder(seqs)
        # h = torch.cat((h,features), 1)
        #print(h.shape)
        idx = 0
        for e, seqlen in zip(seq_encode, seqlens):
            if idx in chain_idx:
                seq_embedding = torch.cat((seq_embedding, e[:seqlen].to(self.current_device).reshape( -1, 120)), dim=0)
            idx += 1
        
        pos = 0
        idx = 0
        #print(h.shape)
        single_chain = torch.tensor([]).to(self.current_device).reshape(-1,192*cnt+self.feature_dim*noMid)
        single_features = torch.tensor([]).to(self.current_device).reshape(-1,self.feature_dim)
        for seqlen in seqlens:
            if idx in chain_idx:
                #print(seqlen)
                single_chain = torch.cat((single_chain, h[pos:pos+seqlen].to(self.current_device).reshape( -1, 192*cnt+self.feature_dim*noMid)), dim=0)
                single_features = torch.cat((single_features, features[pos:pos+seqlen].to(self.current_device).reshape( -1, self.feature_dim)), dim=0)
            idx += 1
            pos += seqlen
        
        h = torch.cat((single_chain,seq_embedding), 1)
        # h = torch.cat((h,single_features), 1)
        #noBert
        # h = single_chain
        fixed_features = torch.tensor([]).reshape(-1,self.feature_dim).to(self.current_device)
        pos = 0
        idx = 0
        # for graphlen in  len_graphs:
        #for L in  seqlens:
        #print(seqlens)
        #print(chain_idx)
        for idx in chain_idx:
            
            L = seqlens[idx]

            local_h = single_features[pos:pos+L].to(self.current_device)
            #print(local_h.shape)
            for i in range(L,self.cutoff_len):
                add = torch.tensor([[0 for i in range(self.feature_dim)]]).to(self.current_device)
                local_h = torch.cat((local_h,add), 0)
            if  L > self.cutoff_len:
                local_h = local_h[:self.cutoff_len]
            #print(local_h.shape)
            fixed_features=torch.cat((fixed_features, local_h), 0)
            #print(fixed_features.shape)
            pos += L
        #print(len_graphs)
        #print(fixed_features.shape)
        fixed_features = fixed_features.view(-1,1,self.cutoff_len,self.feature_dim)
        g_features = self.conv2d_1(fixed_features)
        # fixed_features_2 = self.conv2d_2(fixed_features)
        # g_features = self.conv2d_2_pool(fixed_features_2)
        # fixed_features_2 = fixed_features_2.squeeze(3)
        # fixed_features_3 = self.conv2d_3(fixed_features).squeeze(3)
        shapes = g_features.shape
        
        g_features = g_features.view(shapes[0],1,shapes[1])
        # print(g_features.shape)
        # print(shapes)
        # fixed_features_1 = fixed_features_1.view(shapes[0],shapes[2],shapes[1])
        # fixed_features_2 = fixed_features_2.view(shapes[0],shapes[2],shapes[1])
        # fixed_features_3 = fixed_features_3.view(shapes[0],shapes[2],shapes[1])
        # L_features = torch.tensor([]).reshape(-1,shapes[1] * 1).to(self.current_device)
        global_features = torch.tensor([]).reshape(-1, g_features.shape[2]).to(self.current_device)
        # fixed_features = torch.cat((fixed_features_2),2)
        # print(fixed_features[1, :seqlens[idx], :].shape)
        cnt = 0
        # for graphlen in  len_graphs:
        for idx in chain_idx:
            
            # L_features = torch.cat((L_features,fixed_features_2[cnt, :seqlens[idx], :]),0)
            for i in range(seqlens[idx]):
                # print(g_features[cnt].shape)
                global_features = torch.cat((global_features,g_features[cnt]),0)
            cnt += 1


        # fixed_features = self.fc2d1(fixed_features)
        # fixed_features = self.relu(fixed_features)
        # fixed_features = self.dropout1(fixed_features)
        # fixed_features = self.sigmoid(self.fc2d2(fixed_features))
        # # fixed_features = self.relu(fixed_features)
        # #print(fixed_features.shape)
        # idx = 0
        # final_features = torch.tensor([]).to(self.current_device)
        # for graphlen in  len_graphs:
        #     local_h = fixed_features[idx][0:graphlen].to(self.current_device)
        #     idx += 1
        #     final_features=torch.cat((final_features, local_h), 0)
            #print(final_features.shape)
        # final_features = self.sigmoid(final_features)
        # return final_features
        
        # # h = h.unsqueeze(0)
        # pos = 0
        # all_features = []

        # for graphlen in  len_graphs:
        #     sub_h = h[pos:pos+graphlen].to(self.current_device)
        #     sub_h = self.att(sub_h)
        #     #print(sub_h.shape)
        #     all_features.append(sub_h)
        # all_features = torch.cat(all_features ,dim = 0)
        # #print(all_features.shape)
        # #result = self.fc2(all_features)
        # #result = self.sigmoid(result)
        # result = self.mlp(all_features)
        # #print(result.shape)
        # return result

        local_features = torch.tensor([]).reshape(-1,self.feature_dim).to(self.current_device)
        pos = 0
        # for graphlen in  len_graphs:
        for idx in chain_idx:
            L = seqlens[idx]
            local_h = single_features[pos:pos+L].view(1, self.feature_dim, L).to(self.current_device)
           # print(local_h.shape)
            local_h = self.localpool(self.localpad(local_h)).view(L,-1)
            #print(local_h.shape)
            local_features=torch.cat((local_features, local_h), 0)
            pos = pos + L
        # print(local_features.shape)
        # print(global_features.shape)
        # print(h.shape)
        h =  torch.cat((h,local_features,global_features), 1)
        # h =  torch.cat((h,global_features), 1)

        # print(h.shape)
        h = self.DNN1(h)
        h = self.dropout_layer(h)
        h = self.DNN2(h)
        h = self.dropout_layer2(h)
        h = self.outLayer(h)
        #print(h.shape)
        return h

        h = h.unsqueeze(1)

        
        shapes = h.shape
        
        
        #h11 = self.conv11(h)

        #h12 = self.conv12(h)

        #h13 = self.conv13(h) 

        h = self.conv1(h)
        #conv_outs = [h, h11, h12, h13]
        #h = torch.cat(conv_outs, dim=2)
        #print(h.shape)
        #h = self.bn1(h)
        h = self.relu1(h)

        #print(h.shape)
        h = self.pool1(h)
        h = self.conv2(h)
        #print(h.shape)
        #h = self.bn2(h)
        h = self.relu2(h)
        h = self.pool2(h)
        #print(h.shape)
        #h = self.conv2d(h)
        #print(h.shape)
        #shapes = h.data.shape
        h = h.view(shapes[0], -1)
        h = self.fc1(h)
        h = self.relu3(h)
        h = self.dropout1(h)
        h = self.fc2(h)
        h = self.dropout2(h)
        h = self.sigmoid(h)
        print(h.shape)

        return h
        #print(h.shape)
        for i, layer in enumerate(self.classif_layers):
            # if this is the last layer and we want to use a linear layer, the call is different
            if (i == len(self.classif_layers) - 1) and not self.conv_output:
                h = layer(h)
                h = self.sigmoid(h)
            # Convolution layer
            else:
                if self.return_loss:
                    h, loss = layer(g, h)
                    # h = self.relu(h)
                    iso_loss += loss
                else:
                    h = layer(h)
                    h = self.relu(h)

        if self.return_loss:
            return h, iso_loss
        else:
            return h
