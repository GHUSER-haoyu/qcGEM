import torch
import random
import numpy as np
random.seed(1)

def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
        return edges

def MaskProcess(batch_data_dict, mask_token, global_mask_ratio, mask_ratio, replace_ratio, noise_size = 1.0):

    batch_data_dict_label = {}

    batch_data_dict['atom_type'] = torch.nonzero(batch_data_dict['node_features'][:, 32:68].eq(1)).squeeze(dim=1)[:,-1] + torch.tensor(1)

    device_ = batch_data_dict['xyz'].device
    batch_size = batch_data_dict['batch'].max() + 1
    xyz_num = batch_data_dict['xyz'].shape[0]
    atom_num = batch_data_dict['node_features'].shape[0]
    bond_num = batch_data_dict['edge_features'].shape[0]
    mask_ratio = torch.tensor(mask_ratio).to(torch.float32)
    mask_replace_ratio = torch.tensor(replace_ratio).to(torch.float32)
    mask_token_ratio = torch.tensor(1 - replace_ratio).to(torch.float32)

    mask_token_md, mask_token_fp, mask_token_xyz, mask_token_atom, mask_token_bond, mask_token_atomType, _ = mask_token

    batch_data_dict_label['global_MD'] = batch_data_dict['global_MD'].clone().detach().to(device_)
    batch_data_dict_label['global_FP'] = batch_data_dict['global_FP'].clone().detach().to(device_)
    batch_data_dict_label['xyz'] = batch_data_dict['xyz'].clone().detach().to(device_)
    batch_data_dict_label['node_features'] = batch_data_dict['node_features'].clone().detach().to(device_)
    batch_data_dict_label['edge_features'] = batch_data_dict['edge_features'].clone().detach().to(device_)
    batch_data_dict_label['atom_type'] = batch_data_dict['atom_type'].clone().detach().to(device_)

    xyz_feature_index = [[0, 3]]
    atom_feature_index = [[0, 80]]
    bond_feature_index = [[0, 53]]

    mask = torch.bernoulli(torch.full_like(batch_data_dict['global_MD'], global_mask_ratio)).bool()
    batch_data_dict['global_MD'][mask] = mask_token_md[0]
    mask = torch.bernoulli(torch.full_like(batch_data_dict['global_FP'], global_mask_ratio)).bool()
    batch_data_dict['global_FP'][mask] = mask_token_fp[0]

    noise = ((torch.rand_like(batch_data_dict['xyz']) * 2 - 1) * noise_size).to(device_)
    batch_data_dict['xyz'] += noise
    batch_data_dict['pos'] += noise

    for i in range(len(xyz_feature_index)):

        num_mask_xyz = int(mask_ratio * xyz_num)

        xyz_perm = torch.randperm(xyz_num)
        mask_xyz = xyz_perm[: num_mask_xyz]
        masked_atom_index = mask_xyz
        keep_xyz = xyz_perm[num_mask_xyz :]

        if mask_replace_ratio > 0:

            num_token_xyz = int(mask_token_ratio * num_mask_xyz)
            num_noise_xyz = num_mask_xyz - num_token_xyz 

            xyz_perm_mask = torch.randperm(num_mask_xyz)

            token_xyz = mask_xyz[xyz_perm_mask[: num_token_xyz]]
            noise_xyz = mask_xyz[xyz_perm_mask[num_token_xyz:]]

            noise_to_be_chosen = torch.randperm(xyz_num)[: num_noise_xyz]

            batch_data_dict['atom_type'][token_xyz] = torch.tensor(0)
            batch_data_dict['atom_type'][noise_xyz] = batch_data_dict_label['atom_type'][noise_to_be_chosen]
            batch_data_dict['node_features'][token_xyz, :] = torch.tensor(0).to(torch.float32)
            batch_data_dict['node_features'][noise_xyz, :] = batch_data_dict_label['node_features'][noise_to_be_chosen, :]
        else:
            token_xyz = mask_xyz
            batch_data_dict['atom_type'][mask_xyz] = torch.tensor(0)
            batch_data_dict['node_features'][mask_xyz, :] = torch.tensor(0).to(torch.float32)
            
        batch_data_dict['atom_type'][token_xyz] += mask_token_atomType[i]
        batch_data_dict['node_features'][token_xyz, :] += mask_token_atom[i]


    for i in range(len(bond_feature_index)):                            

        num_mask_bond = int(mask_ratio * bond_num)

        bond_perm = torch.randperm(bond_num)
        mask_bond = bond_perm[: num_mask_bond]
        masked_bond_index = mask_bond
        keep_bond = bond_perm[num_mask_bond :]

        if mask_replace_ratio > 0:

            num_token_bond = int(mask_token_ratio * num_mask_bond)
            num_noise_bond = num_mask_bond - num_token_bond 

            bond_perm_mask = torch.randperm(num_mask_bond)

            token_bond = mask_bond[bond_perm_mask[: num_token_bond]]
            noise_bond = mask_bond[bond_perm_mask[num_token_bond:]] 

            noise_to_be_chosen = torch.randperm(bond_num)[: num_noise_bond]

            batch_data_dict['edge_features'][token_bond, :] = torch.tensor(0).to(torch.float32)
            batch_data_dict['edge_features'][noise_bond, :] = batch_data_dict_label['edge_features'][noise_to_be_chosen, :]
        else:
            token_bond = mask_bond
            batch_data_dict['edge_features'][mask_bond, :] = torch.tensor(0).to(torch.float32)
            
        batch_data_dict['edge_features'][token_bond, :] += mask_token_bond[i]
    
    batch_data_dict_label['mask_node_index'] = mask_xyz
    batch_data_dict_label['mask_edge_index'] = mask_bond
    batch_data_dict_label['keep_node_index'] = keep_xyz
    batch_data_dict_label['keep_edge_index'] = keep_bond

    return batch_data_dict, batch_data_dict_label


def local_coord_shoot(FAPE_item, xyz):

    e_w = 1e-8

    pred_num = xyz.shape[0]
    fame_num = FAPE_item.shape[0]
    origin_point_idx = FAPE_item[:, 0]
    x_point_idx = FAPE_item[:, 1]
    y_point_idx = FAPE_item[:, 2]
    

    xyz_tiled = xyz.unsqueeze(1).tile(1, fame_num, 1, 1)
    
    xyz_ = xyz.clone()
    xyz_origin_points = xyz_[:, origin_point_idx.type(torch.long), :]
    xyz_x_points = xyz_[:, x_point_idx.type(torch.long), :]
    xyz_y_points = xyz_[:, y_point_idx.type(torch.long), :]
    xyz_x_direct = xyz_x_points - xyz_origin_points
    xyz_y_direct = xyz_y_points - xyz_origin_points

    xyz_z_bar_local = torch.cross(xyz_x_direct, xyz_y_direct, dim = -1)
    xyz_y_bar_local = torch.cross(xyz_z_bar_local, xyz_x_direct, dim = -1)
    xyz_x_bar_local = xyz_x_direct

    xyz_tiled_Translated = xyz_tiled - xyz_origin_points.reshape(pred_num, fame_num, 1, 3)

    shoot2local_xyz_x = (torch.matmul(xyz_tiled_Translated, xyz_x_bar_local.reshape(pred_num, fame_num, 3, 1)) / (torch.linalg.norm(xyz_x_bar_local, dim = -1).reshape(pred_num, fame_num, 1, 1) + e_w))
    shoot2local_xyz_y = (torch.matmul(xyz_tiled_Translated, xyz_y_bar_local.reshape(pred_num, fame_num, 3, 1)) / (torch.linalg.norm(xyz_y_bar_local, dim = -1).reshape(pred_num, fame_num, 1, 1) + e_w))
    shoot2local_xyz_z = (torch.matmul(xyz_tiled_Translated, xyz_z_bar_local.reshape(pred_num, fame_num, 3, 1)) / (torch.linalg.norm(xyz_z_bar_local, dim = -1).reshape(pred_num, fame_num, 1, 1) + e_w))
    
    xyz_tiled_Translated_Rotated = torch.concat((shoot2local_xyz_x, shoot2local_xyz_y, shoot2local_xyz_z), axis = -1)
    
    return xyz_tiled_Translated_Rotated

def align_GPU(mol_label, mol_pred, device_) :
    batch_num = mol_label.size(0)
    
    label_center_point = torch.mean(mol_label, dim=-2, keepdim=True)
    pred_center_point = torch.mean(mol_pred, dim=-2, keepdim=True)
    
    mol_label_removedC = mol_label - label_center_point
    mol_pred_removedC = mol_pred - pred_center_point

    H = torch.matmul(mol_label_removedC.transpose(dim0=-2, dim1=-1) , mol_pred_removedC)

    U, S, Vh = torch.linalg.svd(H)
    V = Vh.mH

    R = torch.matmul(V, U.transpose(dim0=-2, dim1=-1))

    # Remove Reflect
    det = torch.linalg.det(R)
    eye = torch.eye(3).tile(batch_num, 1, 1).to(device_)
    eye[:, 2, 2] = det
    R = torch.matmul(V, eye).matmul(U.transpose(dim0=-2, dim1=-1))

    T = -torch.matmul(R, label_center_point.transpose(dim0=-2, dim1=-1)) + pred_center_point.transpose(dim0=-2, dim1=-1)
    
    mol_label = torch.matmul(mol_label, R.transpose(dim0=-2, dim1=-1)) + T.transpose(dim0=-2, dim1=-1)
    
    return mol_label



#####============================================================================>>>>>>>>>>>>>>>>>>>>>>>>>
##### functions for models
#####============================================================================>>>>>>>>>>>>>>>>>>>>>>>>>

def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    
    data_device = data.device
    segment_ids = segment_ids.to(data_device)
    
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  
    count = data.new_full(result_shape, 0)
    
    data_device = data.device
    segment_ids = segment_ids.to(data_device)
    
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

def coord2radial(edge_index, coord):
    row, col = edge_index
    coord_diff = coord[row] - coord[col]
    radial = torch.sum(coord_diff**2, -1).unsqueeze(-1)
    return radial, coord_diff

def node2node_diff(edge_index, node):

    row, col = edge_index
    node_diff = node[col] - node[row]
    node_distance = torch.sum(node_diff**2, 1).unsqueeze(1)
    
    norm_col_safe = torch.norm(node[col], dim=1).clamp(min=1e-12) 
    norm_row_safe = torch.norm(node[row], dim=1).clamp(min=1e-12)   
    dot_product = (node[col] * node[row]).sum(dim=1)  
    cosine_similarity = dot_product / (norm_col_safe * norm_row_safe)
    cosine_similarity = cosine_similarity.unsqueeze(-1)
    
    node_diff = torch.concat([node_diff, node_distance, cosine_similarity], dim = -1)
    return node_diff

def Vec_Dis_batch_loss(coord, batch_clip_size):

    edge_index = get_edges_(batch_clip_size)
    row, col = edge_index
    vec = coord[:, :, row, :] - coord[:, :, col, :]
    paired_distance = torch.norm(vec, p = 2, dim = -1).unsqueeze(-1)
    return vec, paired_distance

def get_edges_(n_nodes):
    indices = torch.cartesian_prod(torch.arange(n_nodes), torch.arange(n_nodes))
    return indices.T.tolist()
