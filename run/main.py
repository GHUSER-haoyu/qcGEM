"""
Import packages
"""
import os
import sys
import argparse
import random
import time
import numpy as np
import pandas as pd
import copy
import torch
import torch.utils.data
from torch import nn, optim
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import torch.nn.init as init  

import models
from dataset_pyg import qcGEM_Data
from cal_loss import Loss_GNE

"""
Setting the args
"""
parser = argparse.ArgumentParser(description='Args of Representation Pre-train Model')

parser.add_argument('--encoder_method', type=str, default='qcGEM_Encoder', metavar='N',
                    help='Select the encoder model.')
parser.add_argument('--decoder_method', type=str, default='qcGEM_Decoder', metavar='N',
                    help='Select the decoder model.')
parser.add_argument('--encoder_layers', type=int, default=16, metavar='N',
                    help='The number of layers for the encoder.')
parser.add_argument('--decoder_layers', type=int, default=0, metavar='N',
                    help='The number of layers for the decoder.')
parser.add_argument('--heads', type=int, default=8, metavar='N',
                    help='The number of attention heads.')
parser.add_argument('--global_head_dim', type=int, default=32, metavar='N',
                    help='Head dimension of global.')
parser.add_argument('--node_head_dim', type=int, default=32, metavar='N',
                    help='Head dimension of node.')
parser.add_argument('--edge_head_dim', type=int, default=32, metavar='N',
                    help='Head dimension of edge.')
parser.add_argument('--botnec_global_dim', type=int, default=128, metavar='N',
                    help='Bottleneck dimension of global.')
parser.add_argument('--botnec_node_dim', type=int, default=128, metavar='N',
                    help='Bottleneck dimension of node.')
parser.add_argument('--botnec_edge_dim', type=int, default=128, metavar='N',
                    help='Bottleneck dimension of edge.')
parser.add_argument('--gm_interact_time', type=int, default=4, metavar='N',
                    help='The number of structure model interaction times.')
parser.add_argument('--gm_layer_num', type=int, default=3, metavar='N',
                    help='The number of structure model output layers.')
parser.add_argument('--gm_cutoff', type=float, default=8.0, metavar='N',
                    help='Local structure cutoff.')
parser.add_argument('--gm_output_dim', type=int, default=12, metavar='N',
                    help='Structure model output dimension.')
parser.add_argument('--init', type=str, default='None', metavar='N',
                    help='Weight init method.')
parser.add_argument('--norm', type=str, default='layer', metavar='N',
                    help='The normalization method.')

parser.add_argument('--mask_method', type=str, default='Mask_Replace_Noise', metavar='N',
                    help='Select mask methods.')
parser.add_argument('--global_mask_ratio', type=float, default=0.0, metavar='N',
                    help=' You should set a mask ratio of global state.')
parser.add_argument('--mask_ratio', type=float, default=0.0, metavar='N',
                    help=' You should set a mask ratio of local information.')
parser.add_argument('--replace_ratio', type=float, default=0.0, metavar='N',
                    help=' You should set a replace ratio of local information.')
parser.add_argument('--remask_ratio', type=float, default=0.0, metavar='N',
                    help=' You should set a remask ratio of local information.')
parser.add_argument('--remove_self_loop', action='store_true', default=False,
                    help='Remove the self loop.')

parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='The number of epochs.')
parser.add_argument('--pretrain_set', type=str, default='20250101', metavar='N',
                    help='Pre-training dataset.')
parser.add_argument('--pretrain_set_split_method', type=str, default='random', metavar='N',
                    help='How to split the pre-training dataset.')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='The training batch size.')
parser.add_argument('--shuffle', action='store_true', default=False,
                    help='Shuffle or not.')
parser.add_argument('--lr', type=float, default=1e-4, metavar='N',
                    help='The learning rate.')
parser.add_argument('--lr_method', type=str, default='None', metavar='N',
                    help='The learning rate decay method.')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Use pretrained model or not.')
parser.add_argument('--use_FAPE', action='store_true', default=False,
                    help='Use FAPE loss or Not.')
parser.add_argument('--use_grad_clip', action='store_true', default=False,
                    help='Use grad clip or not.')

parser.add_argument('--log_file_name', type=str, default='None', metavar='N',
                    help='The name of log file.')
parser.add_argument('--no_cpu', action='store_true', default=False,
                    help='Use cpu or gpu.')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='The gpu device number.')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='Random seed.')

parser.add_argument('--root_path', type=str, default='../data/', metavar='N',
                    help='The root path of dataset.')
parser.add_argument('--fape_path', type=str, default='../data/fape.npz', metavar='N',
                    help='The path of FAPE npy file.')
parser.add_argument('--save_path', type=str, default='../model', metavar='N',
                    help='The path to save the model.')
parser.add_argument('--log_file_path', type=str, default='../log_file', metavar='N',
                    help='The path to save the model.')
parser.add_argument('--pretrained_path', type=str, default='../model', metavar='N',
                    help='The path of pretrained model.')
parser.add_argument('--pretrained_model', type=str, default='qcGEM_ckpt_Index649.pt', metavar='N',
                    help='The name of pretrained model.')

args = parser.parse_args()

"""
Set random seed
"""
def setting(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(" \n ##########   Log the Train and Validation Args   ########## \n")
    if args.log_file_name == 'None':
        args.log_file_name = str(time.perf_counter()).replace('.', '')
        print(f"ReInit the log file name : {args.log_file_name}")
    else:
        print("Use the passed log file name.")

    if args.no_cpu == False:
        args.device = torch.device("cpu")
    elif torch.cuda.is_available():
        args.device = torch.device(args.device)
    else:
        pass

    for k,v in sorted(vars(args).items()):
        print(' ====', k,':',v)

    return args

"""
Loading model
"""
def build_model(args):

    global_dim, xyz_dim, node_dim, edge_dim = [200, 512, 512], 3, 80, 53

    model = models.qcGEM(input_global_dim= global_dim, global_head_dim = args.global_head_dim, botnec_global_dim = args.botnec_global_dim, 
                    input_node_dim = node_dim, node_head_dim = args.node_head_dim, BotNec_node_dim = args.botnec_node_dim, 
                    input_edge_dim = edge_dim, edge_head_dim = args.edge_head_dim, BotNec_edge_dim = args.botnec_edge_dim,
                    heads = args.heads, 
                    device = args.device, act_fn = nn.GELU(), norm = args.norm,
                    remove_self_loop = args.remove_self_loop,
                    global_mask_ratio = args.global_mask_ratio, mask_ratio = args.mask_ratio, replace_ratio = args.replace_ratio, remask_ratio = args.remask_ratio,
                    encoder_method = args.encoder_method, decoder_method = args.decoder_method,
                    encoder_layers = args.encoder_layers, decoder_layers = args.decoder_layers,
                    gm_cutoff = args.gm_cutoff, gm_output_dim = args.gm_output_dim, gm_interact_time = args.gm_interact_time, gm_layer_num = args.gm_layer_num)

    num_1 = sum(p.numel() for p in model.parameters())
    num_2 = sum(p.numel() for p in model.encoder.parameters())
    num_3 = sum(p.numel() for p in model.decoder.parameters())
    print(f' ==== The total num of parameters is {num_1}, Encoder is {num_2}, Decoder is {num_3}.')

    if args.pretrained == True:
        pre_trained_model_state = torch.load(f'{args.pretrained_path}/{args.pretrained_model}', map_location=torch.device(args.device))
        model.load_state_dict(pre_trained_model_state['model_state_dict'], strict=True)
    else:
        pass

    return args, model

"""
Loading optimizer
"""
def build_optimizer(args, model):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    return args, optimizer

"""
Data loading
"""
def build_dataset(args):
    fape = np.load(args.fape_path, allow_pickle=True)
    DataSet = qcGEM_Data(root = args.root_path, dataset = args.pretrain_set, split_mode = args.pretrain_set_split_method, split_seed = args.seed)
    data_loader_train = DataLoader(DataSet.train, batch_size=args.batch_size, shuffle = args.shuffle)
    data_loader_valid = DataLoader(DataSet.val, batch_size=args.batch_size, shuffle = False)

    return args, data_loader_train, data_loader_valid, fape

"""
Traning and validation functions
"""
def train(args, epoch, loader, loss_logger, fape):
    total_grad = 0
    loss_logger = {key: 0 for key in loss_logger.keys()}
    loss_logger['epoch_num'] = epoch
    loss_logger['pattern'] = 'train'
    loss_logger['sample_times'] = len(loader) * 1
    model.train()
    for i, batch_data in enumerate(loader):
        batch_data.to(args.device)        
        pred_result, label, _ = model(batch_data)
        Loss_Dict = Loss_GNE(pred_result, label, FAPE, args.use_FAPE)
        Loss_Dict['Total_Loss'].backward()
        if args.use_grad_clip:
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=10)
        else:
            pass
        for key in list(loss_logger.keys())[4:]:
            if key in ['FAPE_Loss', 'Distance_Loss', 'Vector_Loss', 'RMSD_Loss']:
                loss_logger[key] += Loss_Dict[key].detach().cpu().numpy()
            else:
                loss_logger[key] += Loss_Dict[key].item()
        optimizer.step()
        optimizer.zero_grad()
    return loss_logger

def valid(args, epoch, loader, loss_logger, fape):
    loss_logger = {key: 0 for key in loss_logger.keys()}
    loss_logger['epoch_num'] = epoch
    loss_logger['pattern'] = 'valid'
    loss_logger['sample_times'] = len(loader) * 1
    model.eval()
    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            batch_data.to(args.device)        
            pred_result, label, _ = model(batch_data)
            Loss_Dict = Loss_GNE(pred_result, label, FAPE, args.use_FAPE)
            for key in list(loss_logger.keys())[4:]:
                if key in ['FAPE_Loss', 'Distance_Loss', 'Vector_Loss', 'RMSD_Loss']:
                    loss_logger[key] += Loss_Dict[key].detach().cpu().numpy()
                else:
                    loss_logger[key] += Loss_Dict[key].item()
    return loss_logger


"""
Run the training and validation process
"""
def main():

    global args, model, optimizer, data_loader_train, data_loader_valid, FAPE
    args = setting(args)
    args, model = build_model(args)
    args, optimizer = build_optimizer(args, model)
    args, data_loader_train, data_loader_valid, fape = build_dataset(args)

    print(" \n ########## The training and validation process ########## \n")
    train_loss_min = 1000
    valid_loss_min = 1000

    loss_logger = [
                 'epoch_num', 'pattern', 'sample_times', 'timer', 
                 'Total_Loss', 'Total_Loss_global', 'Total_Loss_structure', 'Total_Loss_all_mask', 'Total_Loss_all_keep', 
                 'Total_Loss_local_mask', 'Total_Loss_local_keep', 
                 'MD_Loss', 'FP_Loss', 
                 'FAPE_Loss', 'Distance_Loss', 'Vector_Loss', 'RMSD_Loss',
                 'Atom_RegLoss_mask', 'Atom_ChialTag_Cls_mask', 'Atom_Type_Cls_mask', 'Atom_Hybridization_Cls_mask', 'Atom_IsAromatic_Cls_mask', 'Atom_IsInRing_Cls_mask',
                 'Bond_RegLoss_mask', 'Bond_Stero_Cls_mask', 'Bond_IsConjuated_Cls_mask', 'Bond_IsInRing_Cls_mask', 'Bond_IsAromatic_Cls_mask', 'Bond_Type_Cls_mask', 
                 'Atom_RegLoss_keep', 'Atom_ChialTag_Cls_keep', 'Atom_Type_Cls_keep', 'Atom_Hybridization_Cls_keep', 'Atom_IsAromatic_Cls_keep', 'Atom_IsInRing_Cls_keep',
                 'Bond_RegLoss_keep', 'Bond_Stero_Cls_keep', 'Bond_IsConjuated_Cls_keep', 'Bond_IsInRing_Cls_keep', 'Bond_IsAromatic_Cls_keep', 'Bond_Type_Cls_keep'
                 ]
    LOSS_LOGGER = pd.DataFrame(columns=loss_logger)
    loss_logger = {key: None for key in loss_logger}

    for epoch in range(args.epochs):
        epoch += 1
        print(f'=== Train & Valid Epoch {epoch} ===')

        timer_1 = time.perf_counter()
        loss_logger = train(args, epoch, data_loader_train, loss_logger, fape)
        timer_2 = time.perf_counter()
        loss_logger['timer'] = timer_2 - timer_1
        epoch_train_loss = loss_logger['Total_Loss'] / loss_logger['sample_times']
        LOSS_LOGGER.loc[len(LOSS_LOGGER)] = loss_logger.values()

        loss_logger = valid(args, epoch, data_loader_valid, loss_logger, fape)
        timer_3 = time.perf_counter()
        loss_logger['timer'] = timer_3 - timer_2
        epoch_valid_loss = loss_logger['Total_Loss'] / loss_logger['sample_times']
        LOSS_LOGGER.loc[len(LOSS_LOGGER)] = loss_logger.values()

        LOSS_LOGGER.to_csv(f'{args.log_file_path}/LOGGER_{args.log_file_name}.csv')

        if train_loss_min > epoch_train_loss and valid_loss_min > epoch_valid_loss:
            train_loss_min = epoch_train_loss
            valid_loss_min = epoch_valid_loss
            Save_model_time = time.perf_counter()
            print(f'Epoch {epoch} has the best training and validation loss : {epoch_train_loss} / {epoch_valid_loss}')
            print(f'Save the model : Epo_{epoch}_Seed_{args.seed}_T{Save_model_time}.pt')
            sys.stdout.flush()
            torch.save({'epochs':epoch, 'model_state_dict':model.state_dict(), 'optimizer_state_dict':optimizer.state_dict()}, 
            args.save_path + f'/Epo_{epoch}_Seed_{args.seed}_T_{Save_model_time}.pt')
        else:
            print(f'Epoch {epoch} not the best, the loss is {epoch_train_loss} / {epoch_valid_loss}')
            sys.stdout.flush()

"""
main
"""
if __name__ == "__main__":
    main()



