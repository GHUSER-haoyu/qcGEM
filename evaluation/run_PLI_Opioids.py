"""
Import packages
"""
import sys
import argparse
import random
import time
import numpy as np
import datetime

import torch
import torch.utils.data
from torch import nn, optim
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything

import models
from dataset_pyg import PLI_Opioids_Dataset
import cal_loss
import datetime
from sklearn.metrics import roc_auc_score

"""
Setting the args
"""
parser = argparse.ArgumentParser(description='PLI Opioids Downstream Task')
parser.add_argument('--embed_type', type=str, default='Hard', metavar='N',
                    help='emebdding type')
parser.add_argument('--dataset', type=str, default='CYP2D6_reg', metavar='N',
                    help='dataset name')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size')
parser.add_argument('--shuffle', type=bool, default=True, metavar='N',
                    help='shuffle or not')
parser.add_argument('--optim', type=str, default='Adam', metavar='N',
                    help='optimizer type')
parser.add_argument('--learning_methods', type=str, default='No', metavar='N',
                    help='learning methods')
parser.add_argument('--lr', type=float, default=1e-4, metavar='N',
                    help='learning rate')
parser.add_argument('--c_loss', type=str, default='ce',
                    help='classification loss')
parser.add_argument('--r_loss', type=str, default='rmse',
                    help='regression loss')
parser.add_argument('--norm', type=str, default='layer',
                    help='normlization method')

parser.add_argument('--encoder_method', type=str, default='qcGEM_Encoder', metavar='N',
                    help='encoder name')
parser.add_argument('--decoder_method', type=str, default='qcGEM_Decoder', metavar='N',
                    help='decoder name')
parser.add_argument('--predictor', type=str, default='PLI_opioids_probes', metavar='N',
                    help='downstream predictor name')

parser.add_argument('--global_mask_ratio', type=float, default=0.0,
                    help='mask ratio')
parser.add_argument('--mask_ratio', type=float, default=0.0,
                    help='mask ratio')
parser.add_argument('--remask_ratio', type=float, default=0.0,
                    help='remask ratio')
parser.add_argument('--replace_ratio', type=float, default=0.0,
                    help='replace ratio')
parser.add_argument('--global_dim', type=int, default=128, metavar='N',
                    help='global feature dimension')
parser.add_argument('--node_dim', type=int, default=128, metavar='N',
                    help='node feature dimension')
parser.add_argument('--edge_dim', type=int, default=128, metavar='N',
                    help='edge feature dimension')
parser.add_argument('--remove_self_loop', action='store_true', default=False, 
                    help='remove self loop or not')
parser.add_argument('--use_pretrained', action='store_true', default=False, 
                    help='use pretrained model or not')
parser.add_argument('--pretrained_path', type=str, default='../model', metavar='N',
                    help='The path of pretrained model.')
parser.add_argument('--pretrained_model', type=str, default='qcGEM_ckpt_Index649.pt', metavar='N',
                    help='The name of pretrained model.')

parser.add_argument('--no_cpu', action='store_true', default=False,
                    help='do not use cpu')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='which device to use for training')
parser.add_argument('--seed', type=int, default=0, metavar='N',
                    help='random seed')
parser.add_argument('--pataience', type=int, default=10, metavar='N',
                    help='pataience for early stop')

parser.add_argument('--root_path', type=str, default='../data/evaluation/PLI/', metavar='N',
                    help='dataset root path')
parser.add_argument('--data_split_mode', type=str, default='random',
                    help='split method for dataset')
    

args = parser.parse_args()


def setting(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.determinstic = True
    seed_everything(args.seed)

    if args.no_cpu == False:
        args.device = torch.device("cpu")
    elif torch.cuda.is_available():
        args.device = torch.device(args.device)
    else:
        pass

    loss_function = cal_loss.Regression_loss

    for k,v in sorted(vars(args).items()):
        print(' ====', k,':',v)

    return args, loss_function

"""
Loading model
"""
def build_model(args):

    global_dim, xyz_dim, node_dim, edge_dim = [200, 512, 512], 3, 80, 53

    model = models.qcGEM(input_global_dim= global_dim, global_head_dim = 32, botnec_global_dim = args.global_dim, 
                    input_node_dim = node_dim, node_head_dim = 32, BotNec_node_dim = args.node_dim, 
                    input_edge_dim = edge_dim, edge_head_dim = 32, BotNec_edge_dim = args.edge_dim,
                    heads = 8, 
                    device = args.device, act_fn = nn.GELU(), norm = args.norm,
                    remove_self_loop = args.remove_self_loop,
                    global_mask_ratio = 1.0, mask_ratio = 0.0, replace_ratio = 0.0, remask_ratio = 0.0,
                    encoder_method = 'qcGEM_Encoder', decoder_method = 'qcGEM_Decoder',
                    encoder_layers = 16, decoder_layers = 0,
                    gm_cutoff = 8.0, gm_output_dim = 12, gm_interact_time = 4, gm_layer_num = 3)

    if args.use_pretrained == True:
        pre_trained_model_state = torch.load(f'{args.pretrained_path}/{args.pretrained_model}', map_location=torch.device(args.device))
        model.load_state_dict(pre_trained_model_state['model_state_dict'], strict=True)
    else:
        pass

    if args.predictor == 'PLI_opioids_probes':
        probe_model = models.PLI_opioids_probe(Data_pre_processor = model.pre_process, EmbGenerator = model.encoder, EmbeddingMode = args.embed_type,
                                        global_emb_dim = args.global_dim, node_emb_dim = args.node_dim, edge_emb_dim = args.edge_dim,  
                                        remove_self_loop = args.remove_self_loop,
                                        device = args.device, task_name = args.dataset, norm = args.norm)   
    else:
        print('Type in the right predictor name .')

    return args, probe_model

"""
Loading optimizer
"""
def build_optimizer(args, model):

    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = 0.01)
    else:
        print('Error optimizer name !')

    if args.learning_methods == 'No':
        lr_scheduler = None
    elif args.learning_methods == 'cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.learning_methods == 'exp':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    else:
        print('Error learning method name !')

    return args, optimizer, lr_scheduler


"""
Data loading
"""
# def build_dataset(args):
#     DataSet = ADMET_Dataset(root = args.root_path, dataset = args.dataset, split_mode = args.data_split_mode, split_seed = args.seed)
#     DataLoader_train = DataLoader(DataSet.train, batch_size=args.batch_size, shuffle = True)
#     DataLoader_val = DataLoader(DataSet.val, batch_size=args.batch_size, shuffle = False)
#     DataLoader_test = DataLoader(DataSet.test, batch_size=args.batch_size, shuffle = False)

#     if args.dataset in ['HalfLife', 'ClearanceHepatocyte', 'ClearanceMicrosome', 'VDss']:
#         from sklearn.preprocessing import MinMaxScaler
#         scaler = MinMaxScaler().fit(DataSet.train.label.reshape(-1, 1))
#         min_ = torch.tensor(scaler.data_min_).to(args.device).to(torch.float32)
#         max_ = torch.tensor(scaler.data_max_).to(args.device).to(torch.float32) 
#         mean_ = min_
#         std_ = max_ - min_
#         preprocess = [min_, max_, mean_, std_]
#     else:
#         preprocess = None

#     return args, DataLoader_train, DataLoader_val, DataLoader_test, preprocess

def build_dataset(args):
    DataSet = PLI_Opioids_Dataset(root = args.root_path, dataset = args.dataset, split_mode = args.data_split_mode, split_seed = args.seed)
    DataLoader_train = DataLoader(DataSet.train, batch_size=args.batch_size, shuffle = True)
    DataLoader_val = DataLoader(DataSet.val, batch_size=args.batch_size, shuffle = False)
    DataLoader_test = DataLoader(DataSet.test, batch_size=args.batch_size, shuffle = False)

    return args, DataLoader_train, DataLoader_val, DataLoader_test

"""
Traning and validation and testing
"""
def train(epoch, loader, loss_function):
    model.train()
    Total_loss = 0
    Total_correct_num = 0
    Total_sample_num = 0
    Total_auc_prediciton = None
    Total_auc_label = None
    Total_MissLabel = None

    for batch_data in loader: 
        batch_data.to(args.device)
        label = batch_data.label.clone().detach()
        batch_num = batch_data.batch.max().clone().detach().cpu().numpy() + 1 
        predict_result = model(batch_data)

        Loss = loss_function(predict_result, label, batch_num, args.label_num, r_loss = args.r_loss)

        Loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        Total_loss += Loss.item()

    sample_times = len(loader)

    print("\n=H= Train Epoch Loss |", epoch + 1, "|", Total_loss / sample_times)

    return Total_loss / sample_times

def valid(epoch, loader, loss_function):
    model.eval()
    with torch.no_grad():
        Total_loss = 0
        Total_correct_num = 0
        Total_sample_num = 0
        Total_auc_prediciton = None
        Total_auc_label = None
        Total_MissLabel = None

        for batch_data in loader: 
            batch_data.to(args.device)
            label = batch_data.label.clone().detach()
            batch_num = batch_data.batch.max().clone().detach().cpu().numpy() + 1 
            predict_result = model(batch_data)

            Loss = loss_function(predict_result, label, batch_num, args.label_num, r_loss = args.r_loss)
 
            Total_loss += Loss.item()

    sample_times = len(loader)
    
    print("\n=H= Valid Epoch Loss |", epoch + 1, "|", Total_loss / sample_times)

    return Total_loss / sample_times

def test(epoch, loader, loss_function):

    Total_loss_list = []
    model.eval()
    with torch.no_grad():
        Total_loss = 0
        Total_correct_num = 0
        Total_sample_num = 0
        Total_auc_prediciton = None
        Total_auc_label = None
        Total_MissLabel = None

        for batch_data in loader: 
            batch_data.to(args.device)
            label = batch_data.label.clone().detach()

            batch_num = batch_data.batch.max().clone().detach().cpu().numpy() + 1 
            predict_result = model(batch_data)

            Loss = loss_function(predict_result, label, batch_num, args.label_num, r_loss = args.r_loss)

            Total_loss_list.append(Loss.item()) 
            Total_loss += Loss.item()

    sample_times = len(loader)
    
    print("=Result= Loss | Epoch {}, Avearge is {}, Std is {}".format(epoch + 1, Total_loss / sample_times, np.array(Total_loss_list).std()))

    return Total_loss / sample_times

"""
main
"""
def main():

    global args, model, loss_function, optimizer, data_loader_train, data_loader_valid, data_loader_test

    args, loss_function = setting(args)
    args, model = build_model(args)
    args, optimizer, lr_scheduler = build_optimizer(args, model)
    args, data_loader_train, data_loader_valid, data_loader_test = build_dataset(args)

    print(" \n ########## Log the Train and Validation process ########## \n")
    early_stop = 0
    train_loss_min = 1000
    valid_loss_min = 1000

    for epoch in range(args.epochs):
        epoch += 1
        time_log1 = time.perf_counter()
        sys.stdout.flush()
        print("=====================================================================")
        epoch_train_loss = train(epoch, data_loader_train, loss_function)
        time_log2 = time.perf_counter()
        print('---------------------------------------------------------------------')
        epoch_valid_loss = valid(epoch, data_loader_valid, loss_function)
        time_log3 = time.perf_counter()
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Train timer : ',time_log2 - time_log1 , 's')
        print('Valid timer : ',time_log3 - time_log2 , 's')
        print('Total timer : ',time_log3 - time_log1 , 's')

        if lr_scheduler is not None:
            lr_scheduler.step()
        else:
            pass

        if train_loss_min > epoch_train_loss and valid_loss_min > epoch_valid_loss:
            train_loss_min = epoch_train_loss
            valid_loss_min = epoch_valid_loss
            # torch.save(model.state_dict(), f'../model/ADMET_{args.dataset}_{args.data_split_mode}_{args.seed}.ckpt')
            print("Saved in epoch :", str(epoch))
            print("The best training and validation loss: ", train_loss_min, valid_loss_min)
            print('*******************\n')
            test(epoch, data_loader_test, loss_function)
            print('*******************\n')
            early_stop = 0
        else:
            early_stop += 1

        if early_stop >= args.pataience: 
            print('Early stop the task')

        if early_stop >= (args.pataience * 2):
            print('Double early stop the task')

"""
main
"""
if __name__ == "__main__":
    main()
