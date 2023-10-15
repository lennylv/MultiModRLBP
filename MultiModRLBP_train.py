import os
import sys
import argparse
import torch
import random
import pickle
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', '..'))
sys.path.append('RnaBert')
from Model import model, learn

from RGCN.data_loading import graphloader
from RGCN.benchmark import evaluate
from RGCN.kernels import node_sim

from RnaBert.MLM_SFP import  get_config, TRAIN, BertModel, BertForMaskedLM

node_target = ['binding_small-molecule']
node_features = [ 'nt_code', 'alpha', 'beta', 'gamma', 'delta', 
                      'epsilon', 'zeta', 'epsilon_zeta',  'chi',  'C5prime_xyz',
                        'P_xyz', 'ssZp', 'Dp', 'splay_angle', 'splay_distance', 'splay_ratio', 
                        'eta', 'theta', 'eta_prime', 'theta_prime', 'eta_base', 'theta_base', 'v0', 'v1',
                          'v2', 'v3', 'v4', 'amplitude', 'phase_angle',  
                        'suiteness', 'filter_rmsd','puckering','sugar_class','bin','TotalAsa','PolarAsa','ApolarAsa']
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="trainParams")
    parser.add_argument("--epochs", type=int, default=80, help="the epochs of model will train")
    parser.add_argument("--save_path", type=str, default="save_model.pth", help="the model save path")
    parser.add_argument("--load_path", type=str, default="save_model.pth", help="the model to load")
    parser.add_argument("--train_batch", type=int, default=5, help="train_batch")
    parser.add_argument("--test_batch", type=int, default=1, help="test_batch")
    parser.add_argument("--testSet", type="test18", default=1, help="testSet")
    args = parser.parse_args()

    # graph_index = pickle.load(open('graph_index_440.json','rb'))

    AUC = []
    MCC = []
    P = []
    R = []

    cl3 = ['1uts_b','1uui_b','2l8h_a','1arj_n','1uud_b']
    # remove_cl = [0, 3, 19, 30, 23, 89, 5, 11, 13, 80, 108, 8, 88, 126, 128, 129,  117, 75, 156, 173]
    #remove_cl = [0, 13, 5, 11, 80, 108, 8, 88, 126, 128, 129, 8, 117, 75, 156, 173]
    remove_cl = []
    train_clstr = [x  for x in range(180) if x not in remove_cl]

    #train_clstr = [x  for x in range(180)]
    train_split = []
    test_split = []
    vaild_split = []
    test_clstr = ['test18']
    #train_clstr= ['train60']
    f = open(f'data/baseline/train60.txt','r')
    data = f.readlines()
    train60 = [line.split('\n')[0].lower()+".json" for line in data if len(line)>=4]
    for x in remove_cl:
        #f = open(f'data/clstr0.3/{x}.clstr','r')
        f = open(f'data/chain_clstr/{x}.clstr','r')
        # f = open(f'data/baseline/{x}.txt','r')
        data = f.readlines()
        rna = [line.split('\n')[0].lower()+".json" for line in data if len(line)>=4]
        #vaild_split.append(rna[0])
        vaild_split.extend(rna)
    for x in train_clstr:
        #f = open(f'data/clstr0.3/{x}.clstr','r')
        f = open(f'data/chain_clstr/{x}.clstr','r')
        # f = open(f'data/baseline/{x}.txt','r')
        data = f.readlines()
        rna = [line.split('\n')[0].lower()+".json" for line in data if len(line)>=4]
        train_split.extend(rna)
    for x in test_clstr:
        #f = open(f'data/clstr/{x}.clstr','r')
        f = open(f'data/baseline/{x}.txt','r')
        data = f.readlines()
        rna = [line.split('\n')[0].lower()+".json" for line in data if len(line)>=4]
        test_split.extend(rna)
    remove_split=['test18']
    test_clstr = ['test18','test9']
    for x in test_clstr:
        #f = open(f'data/clstr/{x}.clstr','r')
        f = open(f'data/baseline/{x}.txt','r')
        data = f.readlines()
        rna = [line.split('\n')[0].lower() for line in data if len(line)>=4]
        remove_split.extend(rna)
    print(remove_split)
    train_split = list(set(train_split))
    #test_split = ['3d2x_a', '3gx2_a', '1y26_x'] 
    for x in train_split:
        #print(x)
        if x in remove_split:
            train_split.remove(x)
            vaild_split.remove(x)
            #print(x)

    for x in train60:
        if x not in train_split :
            # train_split.append(x)
            pass
    for x in train_split:
        if x in test_split:
            train_split.remove(x)
            print(x)
        # if x[0:6] in cl3:
        #     train_split.remove(x)
        #     print(x)
    for x in vaild_split:
        if x in train_split:
            vaild_split.remove(x)
    # train_split.remove('1uud_b.json')
    #train_split = list(set(train_split))
    
    random_seed = 1024
    # 设置随机种子
    seed = 538
    torch.manual_seed(1024)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    #torch.manual_seed(random_seed)
    #random.shuffle(train_split)
    #split_index = int(len(train_split) * 0.2)
    #vaild_split = train_split[:split_index]
    #train_split = train_split[split_index:]
    #vaild_split = []
    vaild_split = test_split
    test3 = ['3d2x_a', '3gx2_a', '1y26_x']
    test4 = ['3d2x_a', '3gx2_a', '1y26_x', '6ez0_a']
    test1 = ['6ez0_a']
    test_split.extend(test4)
    vaild_splitTest3 = ['3d2x_a', '3gx2_a', '1y26_x']
    cl8 = ['1j7t_A','1mwl_A','2be0_A','2bee_A','2et3_A','2et4_A','2et5_A']
    cl19 = ['4tzx_X','4xnr_X','5swe_X']
    cl156 = ['5bjo_E','5bjp_E','6e81_A']
    cl30_1 = ['2gdi_X','4nya_A']
    cl30_2 = ['2hok_A','2hoj_A','2hom_A','2hoo_A','2hop_A','4nyc_A','4nyb_A','4nyd_A','4nyg_A']
    cl1f =['1f1t_a','1fmn_a']
    metal = ['2mis','4pqv','364d','430d']
    nonmetal=['1fmn','379d','2tob','2pwt','4f8u','1f1t','1nem','6ez0','2juk','5v3f','1ddy','1q8n']
    mix =['4yaz','5bjo']
    Pam = ['4zc7_A','1j7t_A','4zc7_B','5zej_B']
    vaild_splitTest18 = [i for i in test_split if i not in  vaild_splitTest3]
    #test_split = ['3d2x_a', '3gx2_a', '1y26_x'] 
    #test_split = ['1ddy_a', '1ddy_c', '1ddy_e','1ddy_g']
    #test_split = ['4f8u_a', '4f8u_b']
    #train_split, test_split = evaluate.get_task_split(node_target=node_target, seed= 1024, graph_index= graph_index)
    print(len(train_split))
    print(test_split)
    print(len(vaild_split))
    print(vaild_splitTest18)

    supervised_train_dataset = graphloader.SupervisedDataset(data_path="data/myData_asa", hashing_path="data/2.5DGraph/iguana/all_graphs_annot_hash.p",
                                                            node_features=node_features,
                                                            redundancy='NR',
                                                            node_target=node_target,
                                                            all_graphs=train_split)
    train_loader = graphloader.GraphLoader(dataset=supervised_train_dataset, split=False, batch_size=args.train_batch).get_data()

    vaild_dataset = graphloader.SupervisedDataset(data_path="data/myData_asa", hashing_path="data/2.5DGraph/iguana/all_graphs_annot_hash.p",
                                                node_features=node_features,
                                                node_target=node_target,
                                                all_graphs=vaild_split)
    vaild_dataset.setNorm(supervised_train_dataset.getNorm())
    vaild_loader = graphloader.GraphLoader(dataset=vaild_dataset, split=False, batch_size=args.test_batch).get_data()
    vaild_datasetTest18 = graphloader.SupervisedDataset(data_path="data/myData_asa", hashing_path="data/2.5DGraph/iguana/all_graphs_annot_hash.p",
                                                node_features=node_features,
                                                node_target=node_target,
                                                all_graphs=vaild_splitTest18)
    vaild_datasetTest18.setNorm(supervised_train_dataset.getNorm())
    vaild_loaderTest18 = graphloader.GraphLoader(dataset=vaild_datasetTest18, split=False, batch_size=args.test_batch).get_data()

    vaild_datasetTest3 = graphloader.SupervisedDataset(data_path="data/myData_asa", hashing_path="data/2.5DGraph/iguana/all_graphs_annot_hash.p",
                                                node_features=node_features,
                                                node_target=node_target,
                                                all_graphs=cl3)
    vaild_datasetTest3.setNorm(supervised_train_dataset.getNorm())
    vaild_loaderTest3 = graphloader.GraphLoader(dataset=vaild_datasetTest3, split=False, batch_size=args.test_batch).get_data()
    #vaild_loader = None
    #RNABert

    config = get_config(file_path = "RNA_bert_config.json")
    config.hidden_size = config.num_attention_heads * config.multiple    
    train = TRAIN(config,device)
    RnaBert = BertModel(config)
    RnaBert = BertForMaskedLM(config, RnaBert)
    RnaBert = train.model_device(RnaBert)
    RnaBert.load_state_dict(torch.load("bert_mul_2.pth"))
    RbertL= 10
    embedder_model = model.RGATEmbedder(infeatures_dim=supervised_train_dataset.input_dim+2 ,
                                        dims=[64, 64])
    # embedder_model_pre.to(device)
    for param in RnaBert.parameters():
        if RbertL < 117: 
            RbertL+=1 
            param.requires_grad = False  # 将所有参数的梯度设置为 False
    # rgcnL_pre = 5
    # for param in embedder_model_pre.parameters():
    #     if rgcnL_pre < 5: 
    #         rgcnL_pre+=1
    #         param.requires_grad = False  # 将所有参数的梯度设置为 False
    rgcnL = 5
    #for param in embedder_model.parameters():
    #  if rgcnL < 5: 
        #    rgcnL+=1
        #   param.requires_grad = False  # 将所有参数的梯度设置为 False
    print(RbertL)
    # Define a model and train it :
    # We first embed our data in 64 dimensions, using the pretrained embedder and then add one classification
    # Then get the training going

    classifier_model = model.RGATClassifier(rgat_embedder=embedder_model, rbert_embedder=RnaBert, 
                                            conv_output=False,
                                            return_loss=False,
                                            classif_dims=[supervised_train_dataset.output_dim])
    classifier_model.deactivate_loss()
    classifier_model.to(device)
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=0.001)
    #model_dict = torch.load(f'/mnt/sdd/user/wjk/MyRNAPredict/crosstest9_0.3/test9_e100_vaild_w11_globallocal_do1.pth')

    #classifier_model.load_state_dict(model_dict['model_state_dict'])
    # learn.train_supervised(model=classifier_model,
    #                 optimizer=optimizer,
    #                  train_loader=train_loader,
    #                   learning_routine=learn.LearningRoutine(num_epochs=args.epochs,device = device, validation_loader= vaild_loader,test3_loader = vaild_loaderTest3,
    # test18_loader = vaild_loaderTest18, save_path =f'/mnt/sdd/user/wjk/MyRNAPredict/crosstest18_0.3/cl3_3.pth'))

    # torch.save(classifier_model.state_dict(), 'final80_cl3_3.pth')

    # embedder_model = models.Embedder(infeatures_dim=4, dims=[64, 64])
    # classifier_model = models.Classifier(embedder=embedder_model, classif_dims=[1])
    test_dataset = graphloader.SupervisedDataset(data_path="data/myData_asa", hashing_path="data/2.5DGraph/iguana/all_graphs_annot_hash.p",
                                                node_features=node_features,
                                                node_target=node_target,
                                                all_graphs=cl1f)
    test_dataset.setNorm(supervised_train_dataset.getNorm())
    test_loader = graphloader.GraphLoader(dataset=test_dataset, split=False, batch_size=1).get_data()
    test_dataset2 = graphloader.SupervisedDataset(data_path="data/myData_asa", hashing_path="data/2.5DGraph/iguana/all_graphs_annot_hash.p",
                                                node_features=node_features,
                                                node_target=node_target,
                                                all_graphs=vaild_split)
    test_dataset2.setNorm(supervised_train_dataset.getNorm())
    test_loader2 = graphloader.GraphLoader(dataset=test_dataset2, split=False, batch_size=1).get_data()
    model_dict = torch.load(args.load_path)

    classifier_model.load_state_dict(model_dict['model_state_dict'])
    # model_dict = torch.load(f'/mnt/sdd/user/wjk/MyRNAPredict/final80_cl3_2.pth')
    # classifier_model.load_state_dict(model_dict)
    # print(test_split)
    # Get a benchmark performance on the official uncontaminated test set :
    from sklearn.metrics import matthews_corrcoef
    # metric, mcc, precision,  recall = evaluate.get_performance(node_target=node_target, node_features=node_features, model=classifier_model,test_loader=test_loader)
    # print('We get a performance of :', metric)
    metric, mcc, precision,  recall = evaluate.get_performance(node_target=node_target, node_features=node_features, model=classifier_model,test_loader=test_loader)
    print('We get a performance of :', metric)
    print()
    AUC.append(metric)
    MCC.append(mcc)
    P.append(precision) 
    R.append(recall)
    for i in range(len(AUC)):
        print(f"AUC : {AUC[i]}  MCC : {MCC[i]}  Precision : {P[i]}  Recall : {R[i]}")



    # Choose the data, features and targets to use
    # node_features = ['nt_code']




    # ###### Unsupervised phase : ######
    # # Choose the data and kernel to use for pretraining
    # print('Starting to pretrain the network')
    # node_sim_func = node_sim.SimFunctionNode(method='R_graphlets', depth=2)
    # unsupervised_dataset = graphloader.UnsupervisedDataset(data_path="data/2.5DGraph/iguana/all_graphs_annot", hashing_path="data/2.5DGraph/iguana/all_graphs_annot_hash.p",
    #                                                        node_simfunc=node_sim_func,
    #                                                        node_features=node_features)
    # # train_loader = graphloader.GraphLoader(dataset=unsupervised_dataset, split=False,
    # #                                        num_workers=0, max_size_kernel=100).get_data()

    # # Then choose the embedder model and pre_train it, we dump a version of this pretrained model
    # embedder_model_pre = model.RGATEmbedder(infeatures_dim=unsupervised_dataset.input_dim ,
    #                                     dims=[64, 64])


    # # optimizer = torch.optim.Adam(embedder_model.parameters())
    # # learn.pretrain_unsupervised(model=embedder_model,
    # #                             optimizer=optimizer,
    # #                             train_loader=train_loader,
    # #                             learning_routine=learn.LearningRoutine(num_epochs=10),
    # #                             rec_params={"similarity": True, "normalize": False, "use_graph": True, "hops": 2})
    # # torch.save(embedder_model.state_dict(), 'pretrained_model_scaled.pth')
    # embedder_model_pre.load_state_dict(torch.load('pretrained_model_scaled.pth'))
    # print()

