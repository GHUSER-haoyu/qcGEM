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
from dataset_pyg import PLI_Dataset
import cal_loss
import datetime
from sklearn.metrics import roc_auc_score

"""
Setting the args
"""
parser = argparse.ArgumentParser(description='PLI Downstream Task')
parser.add_argument('--embed_type', type=str, default='Hard', metavar='N',
                    help='emebdding type')
parser.add_argument('--dataset', type=str, default='FreeSolv', metavar='N',
                    help='dataset name')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size')
parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                    help='number of workers for data loading')
parser.add_argument('--shuffle', type=bool, default=True, metavar='N',
                    help='shuffle or not')
parser.add_argument('--optim', type=str, default='Adam', metavar='N',
                    help='optimizer type')
parser.add_argument('--learning_methods', type=str, default='No', metavar='N',
                    help='learning methods')
parser.add_argument('--lr', type=float, default=1e-4, metavar='N',
                    help='learning rate')
parser.add_argument('--r_loss', type=str, default='rmse',
                    help='regression loss')
parser.add_argument('--c_loss', type=str, default='ce',
                    help='classification loss')
parser.add_argument('--norm', type=str, default='layer',
                    help='normlization method')

parser.add_argument('--encoder_method', type=str, default='qcGEM_Encoder', metavar='N',
                    help='encoder name')
parser.add_argument('--decoder_method', type=str, default='qcGEM_Decoder', metavar='N',
                    help='decoder name')
parser.add_argument('--predictor', type=str, default='PLI_probes', metavar='N',
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
parser.add_argument('--data_split_mode', type=str, default='asPaper',
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

    if args.dataset in ['KIBA', 'DAVIS']:
        loss_function = loss_function_batch.Regression_loss
        args.task_type = 'r'
        args.label_num = 1
    elif args.dataset in ['Human', 'Celegans', 'BindingDB']:                            
        loss_function = nn.BCEWithLogitsLoss()
        args.task_type = 'c'
        args.label_num = 1
    else:
        print('Type in the right dataset name to init the model .')
    

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

    if args.predictor == 'PLI_probes':
        probe_model = models.PLI_probe(Data_pre_processor = model.pre_process, EmbGenerator = model.encoder, EmbeddingMode = args.embed_type,
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
def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    batchC = Batch.from_data_list([data[2] for data in data_list])
    
    return batchA, batchB, batchC
def build_dataset(args):
    DataSet_train = Cliff_Dataset(root=args.root_path, dataset=args.dataset, split_mode='train')
    DataSet_test = Cliff_Dataset(root=args.root_path, dataset=args.dataset, split_mode='test')
    DataLoader_train = DataLoader(DataSet_train, batch_size=args.batch_size, shuffle = True)
    DataLoader_test = DataLoader(DataSet_test, batch_size=args.batch_size, shuffle = False)

    protein_dataset_train = MyDataset_PLI_protein(root=args.root_path, name='protein', split_mode=args.split_mode, mode='train')
    ligand_dataset_train = MyDataset_PLI_ligand(root=args.root_path, name='ligand', split_mode=args.split_mode, mode='train')
    label_dataset_train = MyDataset_PLI_label(root=args.root_path, name='label', split_mode=args.split_mode, mode='train')
    triple_dataset_train = TripleDataset(protein_dataset_train, ligand_dataset_train, label_dataset_train)
    loader_paiered_train = torch.utils.data.DataLoader(triple_dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers, pin_memory=True)

    protein_dataset_valid = MyDataset_PLI_protein(root=args.root_path, name='protein', split_mode=args.split_mode, mode='valid')
    ligand_dataset_valid = MyDataset_PLI_ligand(root=args.root_path, name='ligand', split_mode=args.split_mode, mode='valid')
    label_dataset_valid = MyDataset_PLI_label(root=args.root_path, name='label', split_mode=args.split_mode, mode='valid')
    triple_dataset_valid = TripleDataset(protein_dataset_valid, ligand_dataset_valid, label_dataset_valid)
    loader_paiered_valid = torch.utils.data.DataLoader(triple_dataset_valid, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers, pin_memory=True)

    protein_dataset_test = MyDataset_PLI_protein(root=args.root_path, name='protein', split_mode=args.split_mode, mode='test')
    ligand_dataset_test = MyDataset_PLI_ligand(root=args.root_path, name='ligand', split_mode=args.split_mode, mode='test')
    label_dataset_test = MyDataset_PLI_label(root=args.root_path, name='label', split_mode=args.split_mode, mode='test')
    triple_dataset_test = TripleDataset(protein_dataset_test, ligand_dataset_test, label_dataset_test)
    loader_paiered_test = torch.utils.data.DataLoader(triple_dataset_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers, pin_memory=True)

    return args, loader_paiered_train, loader_paiered_valid, loader_paiered_test

"""
Traning and Testing
"""
def train(epoch, loader, loss_function):
    model.train()
    Total_loss = 0
    all_labels = []
    all_predictions = []

    for batch_data in loader: 
        labels = batch_data[2].label.reshape(-1, 1).to(args.device)
        batch_num = batch_data[1].batch.max().clone().detach().cpu().numpy() + 1 
        predict_result = model(batch_data)
        Loss = loss_function(predict_result, labels)
        Loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        probabilities = torch.sigmoid(predict_result).detach().cpu()
        all_predictions.extend(probabilities.numpy())
        all_labels.extend(list(labels.cpu().numpy().reshape(-1)))

        Total_loss += Loss.item()

    sample_times = len(loader)

    print("\n=H= Train Epoch Loss |", epoch + 1, "|", Total_loss / sample_times)
    if args.task_type == 'c':
        if len(set(all_labels)) > 1:
            auc_score = roc_auc_score(all_labels, all_predictions)
            print("=H= Train Epoch AUC |", epoch + 1, "|", auc_score)
        else:
            print(f"Epoch {epoch+1} - Only one class present in y_true. AUC score is not defined in this case.")
    else:
        pass

    return Total_loss / sample_times

def valid(epoch, loader, loss_function, cliff_loss_function):

    model.eval()
    Total_loss = 0
    all_labels = []
    all_predictions = []
        
    with torch.no_grad():
        for batch_data in loader: 
            labels = batch_data[2].label.reshape(-1, 1).to(args.device)
            batch_num = batch_data[1].batch.max().clone().detach().cpu().numpy() + 1 
            predict_result = model(batch_data)
            Loss = loss_function(predict_result, labels)

            probabilities = torch.sigmoid(predict_result).detach().cpu()
            all_predictions.extend(probabilities.numpy())
            all_labels.extend(list(labels.cpu().numpy().reshape(-1)))

            Total_loss += Loss.item()
            
    sample_times = len(loader)
    
    print("\n=H= Valid Epoch Loss |", epoch + 1, "|", Total_loss / sample_times)
    if args.task_type == 'c':
        if len(set(all_labels)) > 1:
            auc_score = roc_auc_score(all_labels, all_predictions)
            print("=H= Valid Epoch AUC |", epoch + 1, "|", auc_score)
        else:
        print(f"Epoch {epoch+1} - Only one class present in y_true. AUC score is not defined in this case.")
    else:
        pass

    return Total_loss / sample_times


def test(epoch, loader, loss_function, cliff_loss_function):

    model.eval()
    Total_loss = 0
    all_labels = []
    all_predictions = []
        
    with torch.no_grad():
        for batch_data in loader: 
            labels = batch_data[2].label.reshape(-1, 1).to(args.device)
            batch_num = batch_data[1].batch.max().clone().detach().cpu().numpy() + 1 
            predict_result = model(batch_data)
            Loss = loss_function(predict_result, labels)

            probabilities = torch.sigmoid(predict_result).detach().cpu()
            all_predictions.extend(probabilities.numpy())
            all_labels.extend(list(labels.cpu().numpy().reshape(-1)))

            Total_loss += Loss.item()
            
    sample_times = len(loader)
    
    print("\n=H= Test Epoch Loss |", epoch + 1, "|", Total_loss / sample_times)
    if args.task_type == 'c':
        if len(set(all_labels)) > 1:
            auc_score = roc_auc_score(all_labels, all_predictions)
            print("=H= Test Epoch AUC |", epoch + 1, "|", auc_score)
        else:
        print(f"Epoch {epoch+1} - Only one class present in y_true. AUC score is not defined in this case.")
    else:
        pass

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

    print(" \n ########## Log the Train and Test process ########## \n")
    early_stop = 0
    train_loss_min = 1000
    test_loss_min = 1000

    for epoch in range(args.epochs):
        
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
            # torch.save(model.state_dict(), f'../model/PLI_{args.dataset}_{args.data_split_mode}_{args.seed}.ckpt')
            print("Saved in epoch :", str(epoch + 1))
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
