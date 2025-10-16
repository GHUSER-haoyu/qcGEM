import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils import coord2radial, Vec_Dis_batch_loss, local_coord_shoot, align_GPU
from collections import Counter

global_features = np.load('../data/raw/Weight.npz', allow_pickle=True)
fp_weight_NoNorm = torch.from_numpy(global_features['fp_weight'])
fp_weight = (fp_weight_NoNorm + 1e-8) / fp_weight_NoNorm.sum()

def Loss_GNE(pred_result, label, FAPE, use_FAPE):

    Loss_Dict = {}

    mask_atoms_index = label['mask_node_index']
    mask_bonds_index = label['mask_edge_index']

    keep_atoms_index = label['keep_node_index']
    keep_bonds_index = label['keep_edge_index']

    xyz_pred = pred_result['xyz']

    md_pred = pred_result['global_MD']
    fp_pred = pred_result['global_FP']

    atoms_pred = pred_result['node_features']
    bonds_pred = pred_result['edge_features']

    CID_list = pred_result['CID_list']
    edge_index = pred_result['edge_index']
    batch = pred_result['batch']
    ptr_ = pred_result['ptr'].cpu().numpy()

    md_label = label['global_MD']
    fp_label = label['global_FP']
    xyz_label = label['xyz']
    atoms_label = label['node_features']
    bonds_label = label['edge_features']

    """
    Preapare for cal loss
    """
    device_ = xyz_label.device
    batch_size = (batch.max() + 1).detach()
    Pred_XYZ_num = xyz_pred.size(0) 
    xyz_label = xyz_label.unsqueeze(0).tile(Pred_XYZ_num, 1, 1) 

    loss_CE = nn.CrossEntropyLoss(reduction = 'mean') 
    loss_MAE = nn.L1Loss(reduction = 'mean')
    loss_MSE = nn.MSELoss(reduction = 'mean') 
    loss_BCE = nn.BCEWithLogitsLoss(weight = fp_weight.to(device_), reduction = 'mean')

    """
    Loss of Global Features
    """

    Loss_Dict['MD_Loss'] = loss_MAE(md_pred, md_label)
    Loss_Dict['FP_Loss'] = loss_BCE(fp_pred, fp_label)

    """
    Structure Loss:
    FAPE loss + RMSD loss + Global Paired Distance Loss + Global Vec Loss
    """
    FAPE_loss_total = torch.zeros(Pred_XYZ_num).to(torch.float32).to(device_)
    cutoff_num = torch.tensor(5).to(torch.float32).to(device_)
    selected_node = []
    counter = Counter(batch.cpu().numpy())
    batch_clip_size = min(counter.values())
    
    for i in range(batch_size):
        if use_FAPE:
            FAPE_item = torch.tensor(np.array(FAPE[CID_list[i]]['Triangle_Idx_List'])).to(torch.int).to(device_)
            local_coord_pred = local_coord_shoot(FAPE_item, xyz_pred[:, ptr_[i] : ptr_[i+1]]) 
            local_coord_label = local_coord_shoot(FAPE_item, xyz_label[:, ptr_[i] : ptr_[i+1]]) 
            FAPE_loss = torch.norm(local_coord_label - local_coord_pred, p = 2, dim = -1).to(torch.float32) 
            cutoff_ = (torch.ones_like(FAPE_loss) * cutoff_num)
            FAPE_loss = torch.where(FAPE_loss > cutoff_num, cutoff_, FAPE_loss) 
            FAPE_loss = torch.nan_to_num(FAPE_loss, nan = torch.tensor(10).to(torch.float32)).to(device_)
            FAPE_loss_total += torch.mean(FAPE_loss, dim = (1,2))
        else:
            pass

        selected_node_bin = list(np.random.choice(ptr_[i+1] - ptr_[i], size=batch_clip_size, replace=False) + ptr_[i])
        selected_node = selected_node + selected_node_bin

    if use_FAPE:
        Loss_Dict['FAPE_Loss'] = FAPE_loss_total / batch_size
    else:
        Loss_Dict['FAPE_Loss'] = torch.zeros((Pred_XYZ_num)).to(device_)

    selected_node.sort()
    cliped_xyz_pred = xyz_pred[:, selected_node, :].reshape(Pred_XYZ_num * batch_size, batch_clip_size, 3)
    cliped_xyz_label = xyz_label[:, selected_node, :].reshape(Pred_XYZ_num * batch_size, batch_clip_size, 3)
    cliped_xyz_label_RedTed = align_GPU(cliped_xyz_label, cliped_xyz_pred.clone().detach(), device_) # GPU

    cliped_xyz_pred = cliped_xyz_pred.reshape(Pred_XYZ_num, batch_size, batch_clip_size, 3)
    cliped_xyz_label_RedTed = cliped_xyz_label_RedTed.reshape(Pred_XYZ_num, batch_size, batch_clip_size, 3)
    Loss_Dict['RMSD_Loss'] = ((cliped_xyz_pred - cliped_xyz_label_RedTed) ** 2).sum(dim = -1).mean(dim = -1).sqrt().mean(dim = -1)

    Global_Vec_label, Global_DisT_label = Vec_Dis_batch_loss(cliped_xyz_label_RedTed, batch_clip_size)
    Global_Vec_pred, Global_DisT_pred = Vec_Dis_batch_loss(cliped_xyz_pred, batch_clip_size)
    Loss_Dict['Vector_Loss'] = (Global_Vec_label - Global_Vec_pred).abs().sum(dim = -1).mean(dim = -1).mean(dim = -1)
    Loss_Dict['Distance_Loss'] = (Global_DisT_label - Global_DisT_pred).abs().sum(dim = -1).mean(dim = -1).mean(dim = -1)

    # Loss_Dict['RMSD_Loss'] = torch.zeros_like(Loss_Dict['FAPE_Loss']).to(device_)
    # Loss_Dict['Vector_Loss'] = torch.zeros_like(Loss_Dict['FAPE_Loss']).to(device_)
    # Loss_Dict['Distance_Loss'] = torch.zeros_like(Loss_Dict['FAPE_Loss']).to(device_)

    """
    Loss of QM Info
    Atoms and Bond features
    """
    Loss_Dict['Atom_RegLoss_mask'] = loss_MAE(atoms_pred[mask_atoms_index, 0:23], atoms_label[mask_atoms_index, 0:23]) * 23 * 3
    Loss_Dict['Atom_ChialTag_Cls_mask'] = loss_CE(atoms_pred[mask_atoms_index, 23:32], atoms_label[mask_atoms_index, 23:32])
    Loss_Dict['Atom_Type_Cls_mask'] = loss_CE(atoms_pred[mask_atoms_index, 32:68], atoms_label[mask_atoms_index, 32:68])
    Loss_Dict['Atom_Hybridization_Cls_mask'] = loss_CE(atoms_pred[mask_atoms_index, 68:76], atoms_label[mask_atoms_index, 68:76])
    Loss_Dict['Atom_IsAromatic_Cls_mask'] = loss_CE(atoms_pred[mask_atoms_index, 76:78], atoms_label[mask_atoms_index, 76:78])
    Loss_Dict['Atom_IsInRing_Cls_mask'] = loss_CE(atoms_pred[mask_atoms_index, 78:80], atoms_label[mask_atoms_index, 78:80])

    Loss_Dict['Bond_RegLoss_mask'] = loss_MAE(bonds_pred[mask_bonds_index, 0:19], bonds_label[mask_bonds_index, 0:19]) * 19 * 3
    Loss_Dict['Bond_Stero_Cls_mask'] = loss_CE(bonds_pred[mask_bonds_index, 19:25], bonds_label[mask_bonds_index, 19:25])
    Loss_Dict['Bond_IsConjuated_Cls_mask'] = loss_CE(bonds_pred[mask_bonds_index, 25:27], bonds_label[mask_bonds_index, 25:27])
    Loss_Dict['Bond_IsInRing_Cls_mask'] = loss_CE(bonds_pred[mask_bonds_index, 27:29], bonds_label[mask_bonds_index, 27:29])
    Loss_Dict['Bond_IsAromatic_Cls_mask'] = loss_CE(bonds_pred[mask_bonds_index, 29:31], bonds_label[mask_bonds_index, 29:31])
    Loss_Dict['Bond_Type_Cls_mask'] = loss_CE(bonds_pred[mask_bonds_index, 31:53], bonds_label[mask_bonds_index, 31:53])

    Loss_Dict['Atom_RegLoss_keep'] = loss_MAE(atoms_pred[keep_atoms_index, 0:23], atoms_label[keep_atoms_index, 0:23]) * 23 * 3
    Loss_Dict['Atom_ChialTag_Cls_keep'] = loss_CE(atoms_pred[keep_atoms_index, 23:32], atoms_label[keep_atoms_index, 23:32])
    Loss_Dict['Atom_Type_Cls_keep'] = loss_CE(atoms_pred[keep_atoms_index, 32:68], atoms_label[keep_atoms_index, 32:68])
    Loss_Dict['Atom_Hybridization_Cls_keep'] = loss_CE(atoms_pred[keep_atoms_index, 68:76], atoms_label[keep_atoms_index, 68:76])
    Loss_Dict['Atom_IsAromatic_Cls_keep'] = loss_CE(atoms_pred[keep_atoms_index, 76:78], atoms_label[keep_atoms_index, 76:78])
    Loss_Dict['Atom_IsInRing_Cls_keep'] = loss_CE(atoms_pred[keep_atoms_index, 78:80], atoms_label[keep_atoms_index, 78:80])

    Loss_Dict['Bond_RegLoss_keep'] = loss_MAE(bonds_pred[keep_bonds_index, 0:19], bonds_label[keep_bonds_index, 0:19]) * 19 * 3
    Loss_Dict['Bond_Stero_Cls_keep'] = loss_CE(bonds_pred[keep_bonds_index, 19:25], bonds_label[keep_bonds_index, 19:25])
    Loss_Dict['Bond_IsConjuated_Cls_keep'] = loss_CE(bonds_pred[keep_bonds_index, 25:27], bonds_label[keep_bonds_index, 25:27])
    Loss_Dict['Bond_IsInRing_Cls_keep'] = loss_CE(bonds_pred[keep_bonds_index, 27:29], bonds_label[keep_bonds_index, 27:29])
    Loss_Dict['Bond_IsAromatic_Cls_keep'] = loss_CE(bonds_pred[keep_bonds_index, 29:31], bonds_label[keep_bonds_index, 29:31])
    Loss_Dict['Bond_Type_Cls_keep'] = loss_CE(bonds_pred[keep_bonds_index, 31:53], bonds_label[keep_bonds_index, 31:53])
    
    """
    Total Loss
    """

    Loss_Dict['Total_Loss_structure'] = Loss_Dict['FAPE_Loss'].mean() + Loss_Dict['RMSD_Loss'].mean() + Loss_Dict['Distance_Loss'].mean() + Loss_Dict['Vector_Loss'].mean()

    Loss_Dict['Total_Loss_global'] = (Loss_Dict['MD_Loss'] * 200 + Loss_Dict['FP_Loss'] * 512) * 0.25

    Loss_Dict['Total_Loss_local_mask'] = Loss_Dict['Atom_RegLoss_mask'] + Loss_Dict['Atom_ChialTag_Cls_mask'] + Loss_Dict['Atom_Type_Cls_mask'] + Loss_Dict['Atom_Hybridization_Cls_mask'] + Loss_Dict['Atom_IsAromatic_Cls_mask'] + Loss_Dict['Atom_IsInRing_Cls_mask'] \
                                        + Loss_Dict['Bond_RegLoss_mask'] + Loss_Dict['Bond_Stero_Cls_mask'] + Loss_Dict['Bond_IsConjuated_Cls_mask'] + Loss_Dict['Bond_IsInRing_Cls_mask'] + Loss_Dict['Bond_IsAromatic_Cls_mask'] + Loss_Dict['Bond_Type_Cls_mask']
    Loss_Dict['Total_Loss_local_keep'] = Loss_Dict['Atom_RegLoss_keep'] + Loss_Dict['Atom_ChialTag_Cls_keep'] + Loss_Dict['Atom_Type_Cls_keep'] + Loss_Dict['Atom_Hybridization_Cls_keep'] + Loss_Dict['Atom_IsAromatic_Cls_keep'] + Loss_Dict['Atom_IsInRing_Cls_keep'] \
                                        + Loss_Dict['Bond_RegLoss_keep'] + Loss_Dict['Bond_Stero_Cls_keep'] + Loss_Dict['Bond_IsConjuated_Cls_keep'] + Loss_Dict['Bond_IsInRing_Cls_keep'] + Loss_Dict['Bond_IsAromatic_Cls_keep'] + Loss_Dict['Bond_Type_Cls_keep']

    Loss_Dict['Total_Loss_all_mask'] = Loss_Dict['Total_Loss_local_mask']
    Loss_Dict['Total_Loss_all_keep'] = Loss_Dict['Total_Loss_local_keep']

    Loss_Dict['Total_Loss'] = Loss_Dict['Total_Loss_global'] + Loss_Dict['Total_Loss_structure'] + Loss_Dict['Total_Loss_all_mask'] * 1 + Loss_Dict['Total_Loss_all_keep'] * 0.1

    return Loss_Dict



