import torch
from torch import nn
from torch import Tensor
from torch_geometric.nn.norm import GraphSizeNorm, GraphNorm
from torch_geometric.nn import global_add_pool, GENConv, DeepGCNLayer, aggr, GATConv, GATv2Conv, PNAConv, SAGEConv, GINConv, DynamicEdgeConv, MessagePassing, radius_graph
import torch.nn.functional as F
from torch.nn import ReLU, GELU
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.transforms import LineGraph
from torch_geometric.data import Batch
from torch_geometric.utils import remove_self_loops, degree
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (Adj,NoneType, OptTensor, PairTensor, SparseTensor)
from torch_geometric.utils import softmax

import os
import sys
import math
import typing
from typing import Optional, Tuple, Union
if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload_method as overload
from contextlib import ExitStack

from utils import *
from GeometryModule import GeometryModule_

"""
Main Model
"""
class qcGEM(nn.Module):
    def __init__(self, 
                 input_global_dim, global_head_dim, botnec_global_dim, 
                 input_node_dim, node_head_dim, BotNec_node_dim,
                 input_edge_dim, edge_head_dim, BotNec_edge_dim, 
                 heads = None,
                 device=None, act_fn=None, norm=None,
                 remove_self_loop = None, 
                 global_mask_ratio = None, mask_ratio = None, replace_ratio = None, remask_ratio = None,
                 encoder_method = None, decoder_method = None,
                 encoder_layers=None, decoder_layers=None,
                 gm_cutoff = None, gm_output_dim = None, gm_interact_time = None, gm_layer_num = None):
        super(qcGEM, self).__init__()      
        """ 
        init args
        """
        self.heads = heads
        self.input_global_dim = input_global_dim
        self.global_head_dim = global_head_dim
        self.botnec_global_dim = botnec_global_dim
        self.input_node_dim = input_node_dim
        self.node_head_dim = node_head_dim
        self.BotNec_node_dim = BotNec_node_dim
        self.input_edge_dim = input_edge_dim
        self.edge_head_dim = edge_head_dim
        self.BotNec_edge_dim = BotNec_edge_dim

        self.device = device
        self.act_fn = act_fn
        self.norm = norm

        self.encoder_method = encoder_method
        self.decoder_method = decoder_method
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        self.remove_self_loop = remove_self_loop

        self.global_mask_ratio = global_mask_ratio
        self.mask_ratio = mask_ratio
        self.replace_ratio = replace_ratio
        self.remask_ratio = remask_ratio

        self.gm_interact_time = gm_interact_time
        self.gm_layer_num = gm_layer_num
        self.gm_cutoff = gm_cutoff
        self.gm_output_dim = gm_output_dim

        """
        Pre-process
        """
        self.pre_process_method = 'Mask_Replace_XYZnoise'
        if self.pre_process_method == 'Mask_Replace_XYZnoise':
            self.pre_process = PreProcess(global_mask_ratio = self.global_mask_ratio, mask_ratio = self.mask_ratio, replace_ratio = self.replace_ratio, remove_self_loop = self.remove_self_loop, device = self.device, norm = self.norm)
        else:
            print('Error: Please type in a right preprocess name.')
            sys.exit()

        """
        Encoder
        """
        if self.encoder_method == 'qcGEM_Encoder':
            self.encoder = qcGEM_Encoder(
                    input_global_dim = self.input_global_dim, global_head_dim = self.global_head_dim, botnec_global_dim = self.botnec_global_dim,
                    input_node_dim = self.input_node_dim, node_head_dim = self.node_head_dim, BotNec_node_dim = self.BotNec_node_dim,
                    input_edge_dim = self.input_edge_dim, edge_head_dim = self.edge_head_dim, BotNec_edge_dim = self.BotNec_edge_dim,
                    heads = self.heads, 
                    device = self.device, act_fn = self.act_fn, n_layers = self.encoder_layers, norm = self.norm, 
                    gm_cutoff = self.gm_cutoff, gm_output_dim = self.gm_output_dim, gm_interact_time = self.gm_interact_time, gm_layer_num = self.gm_layer_num)
        else:
            print('Error: Please type in a right encoder name.')
            sys.exit()

        """
        Encoder to decoder
        """
        self.encoder2decoder = qcGEM_Encoder2Decoder(mask_ratio = self.mask_ratio, replace_ratio = self.replace_ratio, remask_ratio = self.remask_ratio,
                                               device = self.device, act_fn = self.act_fn, n_layers = None, norm = self.norm)

        """ 
        Decoder
        """
        if self.decoder_method == 'qcGEM_Decoder':
            self.decoder = qcGEM_Decoder(
                    input_global_dim = self.input_global_dim, global_head_dim = self.global_head_dim, botnec_global_dim = self.botnec_global_dim,
                    input_node_dim = self.input_node_dim, node_head_dim = self.node_head_dim, BotNec_node_dim = self.BotNec_node_dim,
                    input_edge_dim = self.input_edge_dim, edge_head_dim = self.edge_head_dim, BotNec_edge_dim = self.BotNec_edge_dim,
                    heads = self.heads,
                    device = self.device, act_fn = self.act_fn, n_layers = self.decoder_layers, norm = self.norm, track_running_stats = True, edge_info_agg = 'sum')  
        else:
            print('Error: Please type in a right decoder name.')
            sys.exit()

        """ 
        Model to device
        """
        self.to(self.device)

    def forward(self, batch_data):
        """
        Pre-process
        """
        batch_data,  batch_data_label = self.pre_process(batch_data)

        """
        Project
        """
        # encoder 2 e2d (project to latent space)
        batch_data, Encoder_Emebdding_Dict = self.encoder(batch_data)        
        # e2d for more operators
        batch_data = self.encoder2decoder(batch_data)
        # e2d 2 deocder (recontruct original features from latent space)
        batch_data = self.decoder(batch_data)

        return batch_data, batch_data_label, Encoder_Emebdding_Dict

"""
Pre-process part
"""
class PreProcess(nn.Module):
    def __init__(self, global_mask_ratio, mask_ratio, replace_ratio, remove_self_loop, device = 'cpu', norm = 'layer'):
        super(PreProcess, self).__init__()

        """
        init args
        """
        self.global_mask_ratio = global_mask_ratio
        self.mask_ratio = mask_ratio
        self.replace_ratio = replace_ratio

        self.remove_self_loop = remove_self_loop
        self.device = device
        self.norm = norm

        """
        Preprocess
        """
        self.Node_QM_LayerNorm = nn.LayerNorm(18, elementwise_affine = False)
        self.Edge_QM_LayerNorm = nn.LayerNorm(19, elementwise_affine = False)
        self.mask_token_md = nn.Parameter(torch.zeros(1,1))
        self.mask_token_fp = nn.Parameter(torch.zeros(1,1))
        self.mask_token_xyz = nn.Parameter(torch.zeros(1, 3))
        self.mask_token_atom_f = nn.Parameter(torch.zeros(1, 80))
        self.mask_token_bond_f = nn.Parameter(torch.zeros(1, 53))
        self.mask_token_atomType = torch.tensor(0)
        self.mask_token_pos = nn.Parameter(torch.zeros(1, 3))
        self.mask_token = (
                        [self.mask_token_md],
                        [self.mask_token_fp],
                        [self.mask_token_xyz],
                        [self.mask_token_atom_f],
                        [self.mask_token_bond_f],
                        [self.mask_token_atomType], 
                        [self.mask_token_pos]
                            )

    def forward(self, batch_data_dict_label):
        
        batch_data_dict_processed = {}
        batch_data_dict_processed['CID_list'] = batch_data_dict_label.CID
        batch_data_dict_processed['global_features'] = None
        batch_data_dict_processed['global_MD'] = batch_data_dict_label.global_MD
        batch_data_dict_processed['global_FP'] = batch_data_dict_label.global_FP
        batch_data_dict_processed['xyz'] = batch_data_dict_label.xyz
        batch_data_dict_processed['pos'] = batch_data_dict_processed['xyz'].clone().detach()
        batch_data_dict_processed['node_features'] = batch_data_dict_label.node_features

        if self.remove_self_loop:
            batch_data_dict_processed['edge_features'] = batch_data_dict_label.edge_features
            batch_data_dict_processed['edge_index'] = batch_data_dict_label.edge_index
            batch_data_dict_processed['edge_index'], batch_data_dict_processed['edge_features'] = remove_self_loops(batch_data_dict_processed['edge_index'], batch_data_dict_processed['edge_features'])
        else:
            batch_data_dict_processed['edge_features'] = batch_data_dict_label.edge_features
            batch_data_dict_processed['edge_index'] = batch_data_dict_label.edge_index

        batch_data_dict_processed['batch'] = batch_data_dict_label.batch
        batch_data_dict_processed['ptr'] = batch_data_dict_label.ptr
        batch_data_dict_processed['num_nodes'] = batch_data_dict_label.num_nodes
        batch_data_dict_processed['graph_task_label'] = None

        batch_data_dict_processed['node_features'][:, :18] = self.Node_QM_LayerNorm(batch_data_dict_processed['node_features'][:, :18])
        batch_data_dict_processed['edge_features'][:, :19] = self.Edge_QM_LayerNorm(batch_data_dict_processed['edge_features'][:, :19])

        batch_data_dict_processed, batch_data_dict_label = MaskProcess(batch_data_dict = batch_data_dict_processed, mask_token = self.mask_token, global_mask_ratio = self.global_mask_ratio, mask_ratio = self.mask_ratio, replace_ratio = self.replace_ratio)

        return batch_data_dict_processed, batch_data_dict_label

"""
Encoder part
"""
class qcGEM_Encoder(nn.Module):
    def __init__(self, 
                 input_global_dim, global_head_dim, botnec_global_dim, 
                 input_node_dim, node_head_dim, BotNec_node_dim,
                 input_edge_dim, edge_head_dim, BotNec_edge_dim, 
                 heads = None,
                 device=None, act_fn=None, n_layers=None, norm=None,
                 gm_cutoff = None, gm_output_dim = None, gm_interact_time = None, gm_layer_num = None):
        super(qcGEM_Encoder, self).__init__()

        """ 
        init args
        """
        self.heads = heads

        self.input_md_dim = input_global_dim[0]
        self.input_fp_dim = input_global_dim[1]
        self.input_global_dim = input_global_dim[2]
        self.global_head_dim = global_head_dim
        self.global_hidden_dim = self.global_head_dim * self.heads
        self.botnec_global_dim = botnec_global_dim

        self.input_node_dim = input_node_dim
        self.node_head_dim = node_head_dim
        self.node_hidden_dim = self.node_head_dim * self.heads
        self.BotNec_node_dim = BotNec_node_dim

        self.input_edge_dim = input_edge_dim
        self.edge_head_dim = edge_head_dim
        self.edge_hidden_dim = self.edge_head_dim * self.heads
        self.BotNec_edge_dim = BotNec_edge_dim

        self.device = device
        self.act_fn = act_fn
        self.n_layers = n_layers
        self.norm = norm

        self.gm_cutoff = gm_cutoff
        self.gm_output_dim = gm_output_dim
        self.gm_interact_time = gm_interact_time
        self.gm_layer_num = gm_layer_num

        """
        Norm method
        """
        if self.norm == 'batch':
            self.node_input_norm = nn.BatchNorm1d(self.node_hidden_dim)
            self.edge_input_norm = nn.BatchNorm1d(self.edge_hidden_dim)
        elif self.norm == 'layer':
            self.node_input_norm = nn.LayerNorm(self.node_hidden_dim)
            self.edge_input_norm = nn.LayerNorm(self.edge_hidden_dim)
        else:
            print('Error Norm Type!')

        """
        Positional embedding
        """
        self.GM_1 = GeometryModule_(cutoff = self.gm_cutoff, num_layers = self.gm_interact_time, num_output_layers = self.gm_layer_num, out_channels = self.gm_output_dim)
        self.GM_2 = GeometryModule_(cutoff = self.gm_cutoff, num_layers = self.gm_interact_time, num_output_layers = self.gm_layer_num, out_channels = self.gm_output_dim)

        self.PE_Atten_bias = nn.Sequential(nn.Linear(1, self.edge_head_dim), self.act_fn, 
                                            nn.Linear(self.edge_head_dim, self.edge_head_dim), self.act_fn, 
                                            nn.Linear(self.edge_head_dim, 1))

        """
        Encoder
        """
        self.embedding_md_input = nn.Sequential(nn.Linear(self.input_md_dim, self.input_global_dim), self.act_fn,
                                                 )
        self.embedding_fp_input = nn.Sequential(nn.Linear(self.input_fp_dim, self.input_global_dim), self.act_fn,
                                                 )
        self.embedding_gf_input = nn.Sequential(nn.Linear(self.input_global_dim * 2, self.global_hidden_dim), self.act_fn,
                                                 nn.Linear(self.global_hidden_dim, self.global_hidden_dim), self.act_fn
                                                 )

        self.embedding_node_input = nn.Sequential(nn.Linear(self.input_node_dim + self.gm_output_dim, self.node_hidden_dim), self.act_fn, 
                                                    nn.Linear(self.node_hidden_dim, self.node_hidden_dim), self.act_fn
                                                    )
        self.embedding_edge_input = nn.Sequential(nn.Linear(self.input_edge_dim, self.edge_hidden_dim), self.act_fn, 
                                                    nn.Linear(self.edge_hidden_dim, self.edge_hidden_dim), self.act_fn
                                                    )

        for i in range(0, self.n_layers):

            self.add_module("Transformer_PlusPE_layer_%d" % i, qcGEM_Transformer(in_channels = self.node_hidden_dim, out_channels = self.node_head_dim, heads = self.heads, concat = True, 
                                                                                        dropout = 0.0, edge_dim = self.edge_hidden_dim, bias = True, root_weight = True))
            
            self.add_module("Local_Readout_%d" %i, aggr.MeanAggregation())
            self.add_module("GLI_%d" % i, GLI_Layer(input_glf_dim=int(self.global_hidden_dim + self.node_hidden_dim), output_glf_dim = self.global_hidden_dim, heads = self.heads, heads_dim = self.global_head_dim, 
                                                    device = self.device, act_fn= self.act_fn, norm = self.norm))

            self.add_module("Global_Update_layer_NecIn_%d" % i, nn.Sequential(nn.Linear(self.global_hidden_dim, self.global_hidden_dim), self.act_fn, 
                                                                                nn.Linear(self.global_hidden_dim, self.botnec_global_dim)))
            self.add_module("Global_Update_layer_NecOut_%d" % i, nn.Sequential(self.act_fn, nn.Linear(self.botnec_global_dim, self.global_hidden_dim), self.act_fn))

            self.add_module("Local_Update_%d" % i, nn.Sequential(nn.Linear(int(self.node_hidden_dim + self.global_hidden_dim), self.node_hidden_dim), self.act_fn))

            self.add_module("Node_Update_layer_NecIn_%d" % i, nn.Sequential(nn.Linear(self.node_hidden_dim + self.gm_output_dim, self.node_hidden_dim), self.act_fn, 
                                                                                nn.Linear(self.node_hidden_dim, self.BotNec_node_dim)))
            self.add_module("Node_Update_layer_NecOut_%d" % i, nn.Sequential(self.act_fn, nn.Linear(self.BotNec_node_dim, self.node_hidden_dim), self.act_fn))
            
            self.add_module("Edge_Update_layer_NecIn_%d" % i, nn.Sequential(nn.Linear(self.edge_hidden_dim + self.node_hidden_dim * 2, self.edge_hidden_dim), self.act_fn, 
                                                                                nn.Linear(self.edge_hidden_dim, self.BotNec_edge_dim)))
            self.add_module("Edge_Update_layer_NecOut_%d" % i, nn.Sequential(self.act_fn, nn.Linear(self.BotNec_edge_dim, self.edge_hidden_dim), self.act_fn))

            if self.norm == 'batch':
                self.add_module("Global_Norm_%d" % i, nn.BatchNorm1d(self.global_hidden_dim + self.node_hidden_dim))
                self.add_module("Node_Norm_%d" % i, nn.BatchNorm1d(self.node_hidden_dim + self.gm_output_dim))
                self.add_module("Edge_Norm_%d" % i, nn.BatchNorm1d(self.edge_hidden_dim + self.node_hidden_dim * 2))
            elif self.norm == 'layer':
                self.add_module("Global_Norm_%d" % i, nn.LayerNorm(self.global_hidden_dim + self.node_hidden_dim))
                self.add_module("Node_Norm_%d" % i, nn.LayerNorm(self.node_hidden_dim + self.gm_output_dim))
                self.add_module("Edge_Norm_%d" % i, nn.LayerNorm(self.edge_hidden_dim + self.node_hidden_dim * 2))
            else:
                print('Error Norm Type!')

        self.embedding_gf_out = nn.Linear(self.botnec_global_dim, self.botnec_global_dim)
        self.embedding_node_out_norm = nn.LayerNorm(self.BotNec_node_dim + self.gm_output_dim)
        self.embedding_node_out = nn.Linear(self.BotNec_node_dim + self.gm_output_dim, self.BotNec_node_dim)
        self.embedding_edge_out = nn.Linear(self.BotNec_edge_dim, self.BotNec_edge_dim)

    def forward(self, batch_data_processed):        

        md = batch_data_processed['global_MD']
        fp = batch_data_processed['global_FP']
        xyz = batch_data_processed['xyz']
        pos = batch_data_processed['pos']
        node_features = batch_data_processed['node_features']
        edge_features = batch_data_processed['edge_features']
        edge_index = batch_data_processed['edge_index']
        atom_type = batch_data_processed['atom_type']
        batch = batch_data_processed['batch']

        LayerInfo_Collect_List = [0, 3, 7, 11, 15]
        GF_State_Embedding = []
        Node_State_Embedding = []
        Edge_State_Embedding = []
        GF_State = []
        Node_State = []
        Edge_State = []

        Node_PE_Stru = self.GM_1(pos, atom_type, batch)
        Node_PE_Stru_ = self.ComENet_2(pos, atom_type, batch)
        Node_PE_Cen = None 
        Edge_PE = xyz[edge_index[0]] - xyz[edge_index[1]] 
        Edge_PE = torch.sum(Edge_PE**2 , 1).unsqueeze(1)
        Edge_PE = self.PE_Atten_bias(Edge_PE).repeat(1, self.heads)

        md = self.embedding_md_input(md)
        fp = self.embedding_fp_input(fp)
        global_features = torch.concat([md, fp], dim=-1)
        global_features = self.embedding_gf_input(global_features)
        node_features = torch.concat([node_features, Node_PE_Stru], dim = -1)
        node_features = self.node_input_norm(self.embedding_node_input(node_features))
        edge_features = self.edge_input_norm(self.embedding_edge_input(edge_features))

        assert self.n_layers == 16

        for i in range(0, self.n_layers):

            """
            Local update
            """
            node_res = node_features
            edge_res = edge_features
            node_features = self._modules["Transformer_PlusPE_layer_%d" % i](x = node_features, edge_index = edge_index, edge_attr = edge_features, 
                                                                            Edge_PE = Edge_PE, return_attention_weights = None)
            node_features = node_res + node_features
            
            node_features = torch.concat([node_features, Node_PE_Stru], dim = -1)
            node_features = self._modules["Node_Norm_%d" % i](node_features)
            node_features = self._modules["Node_Update_layer_NecIn_%d" % i](node_features)

            Node_State_Embedding.append(node_features.clone().detach()) 
            if i in LayerInfo_Collect_List:
                Node_State.append(node_features)
                
            node_features = self._modules["Node_Update_layer_NecOut_%d" % i](node_features)

            """
            Gloal local intearaction 
            """
            lf_features = self._modules["Local_Readout_%d" % i](node_features, index=batch)
            MixFeatures = torch.concat([global_features, lf_features], dim = -1)
            MixFeatures = self._modules["GLI_%d" % i](MixFeatures)
            lf_features = MixFeatures
            gf_features = MixFeatures

            """
            Local update
            """
            node_features = torch.concat([node_features, lf_features[batch]], dim = -1) 
            node_features = self._modules["Local_Update_%d" % i](node_features)
            edge_features = torch.concat([node_features[edge_index[0]], edge_features, node_features[edge_index[1]]], dim = -1)
            edge_features = self._modules["Edge_Norm_%d" % i](edge_features)
            edge_features = self._modules["Edge_Update_layer_NecIn_%d" % i](edge_features)

            Edge_State_Embedding.append(edge_features.clone().detach()) 
            if i in LayerInfo_Collect_List:
                Edge_State.append(edge_features)

            edge_features = self._modules["Edge_Update_layer_NecOut_%d" % i](edge_features)
            edge_features = edge_res + edge_features

            """
            Global update
            """
            global_res = gf_features
            global_features = self._modules["Global_Update_layer_NecIn_%d" % i](gf_features)

            GF_State_Embedding.append(global_features.clone().detach())
            if i in LayerInfo_Collect_List:
                GF_State.append(global_features)

            global_features = self._modules["Global_Update_layer_NecOut_%d" % i](global_features)
            global_features = global_res + global_features

        global_features = torch.stack(GF_State, dim = 0).mean(dim = 0)
        node_features = torch.stack(Node_State, dim = 0).mean(dim = 0)
        edge_features = torch.stack(Edge_State, dim = 0).mean(dim = 0)

        global_features = self.embedding_gf_out(global_features)
        node_features = torch.concat([node_features, Node_PE_Stru_], dim = -1)
        node_features = self.embedding_node_out_norm(node_features)
        node_features = self.embedding_node_out(node_features)
        edge_features = self.embedding_edge_out(edge_features)

        """
        Output
        """
        batch_data_processed['global_features'] = global_features
        batch_data_processed['node_features'] = node_features
        batch_data_processed['edge_features'] = edge_features
        Encoder_Emebdding_Dict = {}
        Encoder_Emebdding_Dict['Global'] = GF_State
        Encoder_Emebdding_Dict['Node'] = Node_State
        Encoder_Emebdding_Dict['Edge'] = Edge_State

        return batch_data_processed, Encoder_Emebdding_Dict


"""
Encoder to decoder
"""
class qcGEM_Encoder2Decoder(nn.Module):
    def __init__(self, input_node_dim = None, NodeDimUps = None, BotNec_node_dim = None,
                 input_edge_dim = None, EdgeDimUps = None, BotNec_edge_dim = None,
                 mask_ratio = None, replace_ratio = None, remask_ratio = None, 
                 device='cpu', act_fn=nn.GELU(), n_layers=None, norm='layer'):
        super(qcGEM_Encoder2Decoder, self).__init__()

        """ 
        init args
        """
        self.input_node_dim = input_node_dim
        self.node_hidden_dim = BotNec_node_dim
        self.BotNec_node_dim = BotNec_node_dim
        self.input_edge_dim = input_edge_dim
        self.edge_hidden_dim = BotNec_edge_dim
        self.BotNec_edge_dim = BotNec_edge_dim

        self.mask_ratio = mask_ratio
        self.replace_ratio = replace_ratio
        self.remask_ratio = remask_ratio

        self.device = device
        self.act_fn = act_fn
        self.n_layers = n_layers
        self.norm = norm

    def forward(self, batch_data_processed):
        
        return batch_data_processed

"""
Decoder part
"""
class qcGEM_Decoder(nn.Module):
    def __init__(self, 
                 input_global_dim, global_head_dim, botnec_global_dim, 
                 input_node_dim, node_head_dim, BotNec_node_dim,
                 input_edge_dim, edge_head_dim, BotNec_edge_dim, 
                 heads = None,
                 device=None, act_fn=None, n_layers=None, norm=None, edge_info_agg = 'sum', track_running_stats = False, PredXYZ_ShareWeight = False):
        super(qcGEM_Decoder, self).__init__()

        """ 
        init args
        """
        self.heads = heads

        self.input_md_dim = input_global_dim[0]
        self.input_fp_dim = input_global_dim[1]
        self.input_global_dim = input_global_dim[2]
        self.global_head_dim = global_head_dim
        self.global_hidden_dim = self.global_head_dim * self.heads
        self.botnec_global_dim = botnec_global_dim

        self.input_node_dim = input_node_dim
        self.node_head_dim = node_head_dim
        self.node_hidden_dim = self.node_head_dim * self.heads
        self.BotNec_node_dim = BotNec_node_dim

        self.input_edge_dim = input_edge_dim
        self.edge_head_dim = edge_head_dim
        self.edge_hidden_dim = self.edge_head_dim * self.heads
        self.BotNec_edge_dim = BotNec_edge_dim

        self.device = device
        self.act_fn = act_fn
        self.n_layers = n_layers
        self.norm = norm
        self.edge_info_agg = edge_info_agg
        self.track_running_stats = track_running_stats
        self.PredXYZ_ShareWeight = PredXYZ_ShareWeight

        """
        Decoder layers
        """
        self.Node_DimChange = nn.Sequential(nn.Linear(self.BotNec_node_dim, self.node_head_dim), self.act_fn
                                            )
        self.Edge_DimChange = nn.Sequential(nn.Linear(self.BotNec_edge_dim, self.edge_head_dim), self.act_fn
                                            )
        self.FO_1 = FoldingOptimizer(input_node_dim = self.node_head_dim, input_edge_dim = self.edge_head_dim, n_layers = self.n_layers, DistanceInfo_tile_num = 3, 
                                            device = self.device, act_fn=nn.GELU(), norm = self.norm, edge_info_agg = self.edge_info_agg)
        self.FO_2 = FoldingOptimizer(input_node_dim = self.node_head_dim, input_edge_dim = self.edge_head_dim, n_layers = self.n_layers, DistanceInfo_tile_num = 3, 
                                            device = self.device, act_fn=nn.GELU(), norm = self.norm, edge_info_agg = self.edge_info_agg)
        self.FO_3 = FoldingOptimizer(input_node_dim = self.node_head_dim, input_edge_dim = self.edge_head_dim, n_layers = self.n_layers, DistanceInfo_tile_num = 3, 
                                            device = self.device, act_fn=nn.GELU(), norm = self.norm, edge_info_agg = self.edge_info_agg)

        self.MD_recovering = nn.Sequential(nn.Linear(self.botnec_global_dim, self.input_global_dim), self.act_fn, nn.Linear(self.input_global_dim, self.input_md_dim))
        self.FP_recovering = nn.Sequential(nn.Linear(self.botnec_global_dim, self.input_global_dim), self.act_fn, nn.Linear(self.input_global_dim, self.input_fp_dim))
        self.QM_recovering_Node = nn.Sequential(nn.Linear(self.BotNec_node_dim, self.input_node_dim), self.act_fn, nn.Linear(self.input_node_dim, self.input_node_dim))                                                   
        self.QM_recovering_Edge = nn.Sequential(nn.Linear(self.BotNec_edge_dim, self.input_edge_dim), self.act_fn, nn.Linear(self.input_edge_dim, self.input_edge_dim))                                                  

    def forward(self, batch_data_processed):

        global_features = batch_data_processed['global_features']
        node_features = batch_data_processed['node_features']
        edge_index = batch_data_processed['edge_index']
        edge_features = batch_data_processed['edge_features']
        atom_type = batch_data_processed['atom_type']
        batch = batch_data_processed['batch']

        """
        Prapare the data
        """
        node_features_forPredXYZ = node_features.clone()
        edge_features_forPredXYZ = edge_features.clone()
        node_features_forRecoverQM = node_features.clone()
        edge_features_forRecoverQM = edge_features.clone()

        """
        Recover info
        """
        md_Recovered = self.MD_recovering(global_features)
        fp_Recovered = self.FP_recovering(global_features)

        node_features_Recovered = self.QM_recovering_Node(node_features_forRecoverQM)
        edge_features_Recovered = self.QM_recovering_Edge(edge_features_forRecoverQM)

        node_features_forPredXYZ = self.Node_DimChange(node_features_forPredXYZ)
        edge_features_forPredXYZ = self.Edge_DimChange(edge_features_forPredXYZ)
        pred_xyz_1, pred_xyz_2, DistMap_1, DistMap_2, node_features_forPredXYZ, edge_features_forPredXYZ = self.FO_1(node_features_forPredXYZ, edge_index, edge_features_forPredXYZ, batch)
        pred_xyz_3, pred_xyz_4, DistMap_3, DistMap_4, node_features_forPredXYZ, edge_features_forPredXYZ = self.FO_2(node_features_forPredXYZ, edge_index, edge_features_forPredXYZ, batch)
        pred_xyz_5, pred_xyz_6, DistMap_5, DistMap_6, node_features_forPredXYZ, edge_features_forPredXYZ = self.FO_3(node_features_forPredXYZ, edge_index, edge_features_forPredXYZ, batch)

        Pred_XYZ = torch.concat([pred_xyz_1.unsqueeze(0), pred_xyz_2.unsqueeze(0), 
                                 pred_xyz_3.unsqueeze(0), pred_xyz_4.unsqueeze(0), 
                                 pred_xyz_5.unsqueeze(0), pred_xyz_6.unsqueeze(0)], dim = 0)

        batch_data_processed['xyz'] = Pred_XYZ
        batch_data_processed['global_MD'] = md_Recovered
        batch_data_processed['global_FP'] = fp_Recovered
        batch_data_processed['node_features'] = node_features_Recovered
        batch_data_processed['edge_features'] = edge_features_Recovered

        return batch_data_processed


"""
qcGEM Transformer
"""
class qcGEM_Transformer(MessagePassing):
    
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    @overload
    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights: NoneType = None,
    ) -> Tensor:
        pass

    @overload
    def forward( 
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Tensor,
        edge_attr: OptTensor = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        pass

    @overload
    def forward( 
        self,
        x: Union[Tensor, PairTensor],
        edge_index: SparseTensor,
        edge_attr: OptTensor = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, SparseTensor]:
        pass

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights: Optional[bool] = None,
        Edge_PE: Tensor = None
    ) -> Union[
            Tensor,
            Tuple[Tensor, Tuple[Tensor, Tensor]],
            Tuple[Tensor, SparseTensor],
    ]:

        H, C = self.heads, self.out_channels
        Edge_PE = Edge_PE

        if isinstance(x, Tensor):
            x = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        out = self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr, Edge_PE=Edge_PE)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int],
                Edge_PE: Tensor) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, 
                                                      self.out_channels)
            key_j = key_j + edge_attr 

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels) 

        alpha = alpha + Edge_PE
        alpha = softmax(alpha, index, ptr, size_i) 
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training) 

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1) 
        
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

"""
Node edge info update
"""
class NEE(nn.Module):
    def __init__(self, input_node_dim, node_hidden_dim, output_node_dim, 
                 input_edge_dim, edge_hidden_dim, output_edge_dim, track_running_stats = False, 
                 act_fn=nn.GELU(), norm = 'layer', edge_info_agg = 'sum'):
        super(NEE, self).__init__()

        """
        init args
        """
        self.edge_info_agg = edge_info_agg
        self.track_running_stats = track_running_stats

        """  
        Edge
        """
        self.edge_mlp = nn.Sequential()
        self.edge_mlp.add_module('edge_Linear1', nn.Linear(input_node_dim * 3 + 1 + 1 + input_edge_dim, edge_hidden_dim))
        if norm == 'batch':
            self.edge_mlp.add_module('edge_BatchNorm1', nn.BatchNorm1d(edge_hidden_dim, track_running_stats = self.track_running_stats))
        elif norm == 'layer':
            self.edge_mlp.add_module('edge_LayerNorm1', nn.LayerNorm(edge_hidden_dim))
        else:
            print("Wrong norm methods")
        self.edge_mlp.add_module('atc1', act_fn)
        self.edge_mlp.add_module('edge_Linear2', nn.Linear(edge_hidden_dim, output_edge_dim))
        if norm == 'batch':
            self.edge_mlp.add_module('edge_BatchNorm2', nn.BatchNorm1d(output_edge_dim, track_running_stats = self.track_running_stats))
        elif norm == 'layer':
            self.edge_mlp.add_module('edge_LayerNorm2', nn.LayerNorm(output_edge_dim))
        else:
            print("Wrong norm methods")
        self.edge_mlp.add_module('atc2', act_fn)

        """
        Node
        """
        self.node_mlp = nn.Sequential()
        self.node_mlp.add_module('node_Linear1', nn.Linear(input_node_dim + output_edge_dim , node_hidden_dim))
        if norm == 'batch':
            self.node_mlp.add_module('node_BatchNorm1', nn.BatchNorm1d(node_hidden_dim, track_running_stats = self.track_running_stats))
        elif norm == 'layer':
            self.node_mlp.add_module('node_LayerNorm1', nn.LayerNorm(node_hidden_dim))
        else:
            print("Wrong norm methods")
        self.node_mlp.add_module('act1', act_fn)
        self.node_mlp.add_module('node_Linear2', nn.Linear(node_hidden_dim, output_node_dim))
        if norm == 'batch':
            self.node_mlp.add_module('node_BatchNorm2', nn.BatchNorm1d(output_node_dim, track_running_stats = self.track_running_stats))
        elif norm == 'layer':
            self.node_mlp.add_module('node_LayerNorm2', nn.LayerNorm(output_node_dim))
        else:
            print("Wrong norm methods")       
        self.node_mlp.add_module('act2', act_fn)

    def edge_model(self, source, target, direct_info, edge_features):
        out = torch.cat([source, target, direct_info, edge_features], dim=1)
        out = self.edge_mlp(out)
        return out
    
    def node_model(self, node_features, edge_index, edge_features):
        row, col = edge_index
        if self.edge_info_agg == 'sum':
            agg = unsorted_segment_sum(edge_features, col, num_segments=node_features.size(0))
        elif self.edge_info_agg == 'mean':
            agg = unsorted_segment_mean(edge_features, col, num_segments=node_features.size(0))
        else:
            raise Exception('Wrong nodes_agg parameter' % self.coords_agg)
        agg = torch.cat([node_features, agg], dim=1)
        out = self.node_mlp(agg)
        return out

    def forward(self, node_features, edge_index, edge_features):
        row, col = edge_index
        node_diff = node2node_diff(edge_index, node_features)
        edge_features = self.edge_model(node_features[row], node_features[col], node_diff, edge_features)
        node_features = self.node_model(node_features, edge_index, edge_features)

        return node_features, edge_features

"""
Global local interaction
"""
class GLI_Layer(nn.Module):
    def __init__(self, input_glf_dim, output_glf_dim, heads, heads_dim, 
                 device = 'cpu', act_fn = nn.GELU(), norm = 'layer'):
        super(GLI_Layer, self).__init__()

        """
        init args
        """
        self.heads = heads
        self.heads_dim = heads_dim
        self.input_glf_dim =input_glf_dim
        self.output_glf_dim = output_glf_dim
        self.hidden_glf_dim = self.heads_dim * self.heads

        self.device = device
        self.act_fn = act_fn
        self.norm = norm

        """
        Gobal local info interaction
        """
        self.pre_process_linear = nn.Sequential(nn.Linear(self.input_glf_dim, self.output_glf_dim), self.act_fn)
        self.token_embedding_linear = nn.Sequential(nn.Linear(1, self.output_glf_dim), self.act_fn)
        self.query_linear = nn.Linear(self.output_glf_dim, self.hidden_glf_dim)
        self.key_linear = nn.Linear(self.output_glf_dim, self.hidden_glf_dim)
        self.value_linear = nn.Linear(self.output_glf_dim, self.hidden_glf_dim)
        self.post_process_linear = nn.Sequential(nn.Linear(self.hidden_glf_dim, self.hidden_glf_dim), self.act_fn, nn.Linear(self.hidden_glf_dim, 1), self.act_fn)

        if norm=="batch":
            self.attention_norm = torch.nn.BatchNorm1d(self.output_glf_dim)
        elif norm=="layer":
            self.attention_norm = torch.nn.LayerNorm(self.output_glf_dim, elementwise_affine=True)
        else:
            print('Normalization strategy is wrong.')

        """
        Feed forward
        """
        self.feed_forward_step_1 = nn.Sequential(nn.Linear(self.output_glf_dim * 1, self.output_glf_dim * 1), self.act_fn)
        self.feed_forward_step_2 = nn.Sequential(nn.Linear(self.output_glf_dim * 1, self.output_glf_dim * 2), self.act_fn,
                                                 nn.Linear(self.output_glf_dim * 2, self.output_glf_dim * 1), self.act_fn)
        if norm=="batch":
            self.feed_forward_norm = torch.nn.BatchNorm1d(self.output_glf_dim)
        elif norm=="layer":
            self.feed_forward_norm = torch.nn.LayerNorm(self.output_glf_dim, elementwise_affine=True)
        else:
            print('Normalization strategy is wrong.')      

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[ : -1] + (self.heads, self.heads_dim)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, glf):

        """
        Attention
        """
        glf = self.pre_process_linear(glf)
        Res = glf
        glf = glf.unsqueeze(-1) 
        glf = self.token_embedding_linear(glf) 

        Query = self.query_linear(glf) 
        Key = self.key_linear(glf) 
        Value = self.value_linear(glf) 

        Query = self.trans_to_multiple_heads(Query) 
        Key = self.trans_to_multiple_heads(Key) 
        Value = self.trans_to_multiple_heads(Value) 

        attention_scores = torch.matmul(Query, Key.permute(0, 1, 3, 2)) / (self.heads_dim ** 0.5) 
        attention_weights = F.softmax(attention_scores, dim = -1) 
        attention_output = torch.matmul(attention_weights, Value).permute(0, 2, 1, 3).contiguous().reshape(-1, self.output_glf_dim, self.hidden_glf_dim) 

        attention_output = self.post_process_linear(attention_output).squeeze(-1)
        attention_output = attention_output + Res
        attention_output = self.attention_norm(attention_output)

        """
        Feed forward
        """
        attention_output = self.feed_forward_step_1(attention_output)
        Res = attention_output
        attention_output = self.feed_forward_step_2(attention_output)
        attention_output = attention_output + Res
        attention_output = self.feed_forward_norm(attention_output)

        return attention_output


"""
Folding optimizer
"""
class FoldingOptimizer(nn.Module):
    def __init__(self, input_node_dim, input_edge_dim, n_layers=None, DistanceInfo_tile_num = 3, track_running_stats = False, 
                 device='cpu', act_fn=nn.GELU(), norm='layer', edge_info_agg = 'sum'):
        super(FoldingOptimizer, self).__init__()
        """ 
        init args
        """
        self.cutoff_bin = torch.tensor([0.01, 0.05, 0.1, 0.12, 0.3, 0.5, 0.8, 1.0, 2.0, 3.0, 5.0, 8.0]).to(device)
        self.node_dim = input_node_dim
        self.edge_dim = input_edge_dim
        self.DistanceInfo_tile_num = self.cutoff_bin.size(0)
        self.edge_distance_dim = self.edge_dim + self.DistanceInfo_tile_num + 1
        self.device = device
        self.act_fn = act_fn
        self.n_layers = n_layers
        self.norm = norm
        self.edge_info_agg = edge_info_agg
        self.track_running_stats = track_running_stats

        """
        Node edge info exchange
        """
        self.N_E_Info_Exchange_1 = NEE(input_node_dim = self.node_dim , node_hidden_dim = self.node_dim * 3, output_node_dim = self.node_dim , 
                                                input_edge_dim = self.edge_dim , edge_hidden_dim = self.edge_dim * 3, output_edge_dim = self.edge_dim,
                                                act_fn=nn.GELU(), norm = self.norm, edge_info_agg = self.edge_info_agg)
        self.N_E_Info_Exchange_2 = NEE(input_node_dim = self.node_dim , node_hidden_dim = self.node_dim * 3, output_node_dim = self.node_dim , 
                                                input_edge_dim = self.edge_dim , edge_hidden_dim = self.edge_dim * 3, output_edge_dim = self.edge_dim,
                                                act_fn=nn.GELU(), norm = self.norm, edge_info_agg = self.edge_info_agg)

        """
        Define PredGeometry model
        """
        self.PG_1 = PredGeometry(input_node_dim = self.node_dim, input_edge_dim = self.edge_dim, n_layers = self.n_layers,
                                       device = self.device, act_fn = self.act_fn, norm = self.norm)
        self.PG_2 = PredGeometry(input_node_dim = self.node_dim, input_edge_dim = self.edge_dim, n_layers = self.n_layers,
                                       device = self.device, act_fn = self.act_fn, norm = self.norm)
        """
        Read the distance info to edge info
        """
        self.DistanceInfo_Process_1 = nn.Sequential()
        self.DistanceInfo_Process_1.add_module('Process_Layer_1', nn.Linear(self.edge_distance_dim, self.edge_distance_dim))
        self.DistanceInfo_Process_1.add_module('Action_Layer_1', self.act_fn)
        self.DistanceInfo_Process_1.add_module('Process_Layer_2', nn.Linear(self.edge_distance_dim, self.edge_dim))
        self.DistanceInfo_Process_1.add_module('Action_Layer_2', self.act_fn)
        self.DistanceInfo_Process_2 = nn.Sequential()
        self.DistanceInfo_Process_2.add_module('Process_Layer_1', nn.Linear(self.edge_distance_dim, self.edge_distance_dim))
        self.DistanceInfo_Process_2.add_module('Action_Layer_1', self.act_fn)
        self.DistanceInfo_Process_2.add_module('Process_Layer_2', nn.Linear(self.edge_distance_dim, self.edge_dim))
        self.DistanceInfo_Process_2.add_module('Action_Layer_2', self.act_fn)

    def forward(self, node_features, edge_index, edge_features, batch):
    
        row, col = edge_index

        node_features_res_1 = node_features.clone()
        edge_features_res = edge_features.clone()
        node_features, edge_features = self.N_E_Info_Exchange_1(node_features, edge_index, edge_features)
        node_features = node_features + node_features_res_1
        pred_xyz_1 = self.PG_1(node_features, edge_index, edge_features)
        xyz_diff = pred_xyz_1[row] - pred_xyz_1[col]
        DistMap_1_ = torch.sum(xyz_diff**2 , 1).unsqueeze(1)
        DistMap_1 = (DistMap_1_.tile(1, self.DistanceInfo_tile_num) < self.cutoff_bin)
        DistMap_1 = torch.concat([DistMap_1, DistMap_1_], dim = -1)
        edge_features = self.DistanceInfo_Process_1(torch.concat([edge_features, DistMap_1], dim = -1))
        edge_features = edge_features + edge_features_res

        node_features_res_2 = node_features.clone()
        edge_features_res = edge_features.clone()
        node_features, edge_features = self.N_E_Info_Exchange_2(node_features, edge_index, edge_features)
        node_features = node_features + node_features_res_1 + node_features_res_2
        pred_xyz_delta = self.PG_2(node_features, edge_index, edge_features)
        pred_xyz_2 = pred_xyz_1 + pred_xyz_delta
        xyz_diff = pred_xyz_2[row] - pred_xyz_2[col]
        DistMap_2_ = torch.sum(xyz_diff**2 , 1).unsqueeze(1)
        DistMap_2 = (DistMap_2_.tile(1, self.DistanceInfo_tile_num) < self.cutoff_bin)
        DistMap_2 = torch.concat([DistMap_2, DistMap_2_], dim = -1)
        edge_features = self.DistanceInfo_Process_2(torch.concat([edge_features, DistMap_2], dim = -1))
        edge_features = edge_features + edge_features_res

        return pred_xyz_1, pred_xyz_2, DistMap_1, DistMap_2, node_features, edge_features

"""
PredGeometry
"""
class PredGeometry(nn.Module):
    def __init__(self, input_node_dim, input_edge_dim, n_layers=None,
                 device='cpu', act_fn=nn.GELU(), norm='layer'):
        super(PredGeometry, self).__init__()

        self.node_dim = input_node_dim
        self.edge_dim = input_edge_dim
        self.n_layers = n_layers
        self.act_fn = act_fn

        self.Predictor_DeeperGCN = nn.ModuleList()
        for i in range(int(self.n_layers)):
            conv = GENConv(self.node_dim, self.node_dim, t = 1.0, learn_t = True, learn_p = True, num_layers = 2, norm = 'layer', edge_dim = self.edge_dim)
            if norm=="batch":
                normalization = torch.nn.BatchNorm1d(self.node_dim)
            elif norm=="layer":
                normalization = torch.nn.LayerNorm(self.node_dim, elementwise_affine=True)
            else:
                print('Wrong normalization strategy!!!')
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, normalization, act, block='res+', dropout=0.1)
            self.Predictor_DeeperGCN.append(layer)

        self.Predictor_MLP = nn.Sequential()
        self.Predictor_MLP.add_module('MLP_Layer_1', nn.Linear(self.node_dim, self.node_dim * 4))
        self.Predictor_MLP.add_module('Normlization', nn.LayerNorm(self.node_dim * 4))
        self.Predictor_MLP.add_module('Action_Layer_1', self.act_fn)
        self.Predictor_MLP.add_module('MLP_Layer_2', nn.Linear(self.node_dim * 4, self.node_dim * 4))
        self.Predictor_MLP.add_module('Action_Layer_2', self.act_fn)
        self.Predictor_MLP.add_module('MLP_Layer_3', nn.Linear(self.node_dim * 4, self.node_dim * 4))
        self.Predictor_MLP.add_module('Action_Layer_3', self.act_fn)
        self.Predictor_MLP.add_module('MLP_Layer_4', nn.Linear(self.node_dim * 4, self.node_dim * 3))
        self.Predictor_MLP.add_module('Action_Layer_4', self.act_fn)
        self.Predictor_MLP.add_module('MLP_Layer_5', nn.Linear(self.node_dim * 3, self.node_dim * 2))
        self.Predictor_MLP.add_module('Action_Layer_5', self.act_fn)
        self.Predictor_MLP.add_module('MLP_Layer_6', nn.Linear(self.node_dim * 2, 3))

    def forward(self, node_features, edge_index, edge_features):
        
        pred_xyz = node_features
        for i, layer in enumerate(self.Predictor_DeeperGCN):
            pred_xyz = layer(pred_xyz, edge_index, edge_features)

        pred_xyz = self.Predictor_MLP(pred_xyz)

        return pred_xyz





# 20250101