import os
import os.path as osp
from itertools import repeat
import pandas as pd

import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset, download_url, Dataset

class MyDataset_batch_hdf5(InMemoryDataset):

    def __init__(self, root, dataset=None, split_mode='random', split_seed=0, transform=None, pre_transform=None, pre_filter=None):
        assert dataset in ['20250101']
        assert split_mode in ['random', 'scaffold']

        self.name = 'PreTrain_data_20250101'
        self.dataset = dataset
        self.split_seed = split_seed
        self.split_mode = split_mode

        super(MyDataset_batch_hdf5, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(osp.join(self.processed_dir, '{}_processed.pt'.format(self.dataset)))
        trn, val = self.split(type = self.split_mode, seed = self.split_seed)
        self.train, self.val = trn, val

    @property
    def raw_dir(self):
        return '/export/disk6/why/workbench/MERGE/DataSet/PreTrain_data_20250101/raw'

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed/{}'.format(self.dataset))

    @property
    def raw_file_names(self):
        name = [ 
        'PubChemQC_PreTrain_{}.hdf5'.format(self.dataset),
        ]
        return name

    @property
    def processed_file_names(self):
        return ['{}_processed.pt'.format(self.dataset)]
    
    def process(self):
        """
        Process the raw data 2 processed data
        """
        data_list = []

        with h5py.File(f'{self.raw_dir}/PubChemQC_PreTrain_{self.dataset}.hdf5', 'r') as all_data_hdf5:

            all_molecular_list = list(all_data_hdf5.keys())

            random.seed(self.split_seed)
            random.shuffle(all_molecular_list)

            i = 0 
            for idx in tqdm(range(len(all_molecular_list)), desc = f'{i+1}/{len(all_molecular_list)}'):
                
                i += 1
                cid = all_molecular_list[idx]
                node_num = int(all_data_hdf5[f'{cid}']['XYZ'].shape[0])
                data = Data()
                data.__num_nodes__ = node_num
                data.CID = cid
                data.smi = all_data_hdf5[f'{cid}']['SMILES'][()].decode('utf-8')
                data.xyz = torch.from_numpy(all_data_hdf5[f'{cid}']['XYZ'][:]).to(torch.float32)
                data.global_MD = torch.from_numpy(np.where(np.isnan(all_data_hdf5[f'{cid}']['Global_MD_norm'][:]), 0, all_data_hdf5[f'{cid}']['Global_MD_norm'][:])).to(torch.float32).reshape(1, -1)
                data.global_FP = torch.from_numpy(np.where(np.isnan(all_data_hdf5[f'{cid}']['Global_FP'][:]), 0, all_data_hdf5[f'{cid}']['Global_FP'][:])).to(torch.float32).reshape(1, -1)
                data.node_features = torch.from_numpy(np.concatenate((all_data_hdf5[f'{cid}']['Node_NBO'][:], all_data_hdf5[f'{cid}']['Node_normal'][:]), axis = 1)).to(torch.float32)
                data.edge_index = torch.from_numpy(np.array(self.get_edges(node_num))).to(torch.int64)
                data.edge_features = torch.from_numpy(np.concatenate((all_data_hdf5[f'{cid}']['Edge_NBO'][:], all_data_hdf5[f'{cid}']['Edge_normal'][:]), axis = 1)).to(torch.float32)
                data.label = None
                data_list.append(data)

        print('making processed files:', self.processed_dir)
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), osp.join(self.processed_dir, '{}_processed.pt'.format(self.dataset)))

    def split(self, type, seed):
        file_path = osp.join(self.processed_dir, '{}_{}_{}.pt'.format(self.dataset, self.split_mode, self.split_seed))
        if os.path.exists(file_path):
            trn, val = torch.load(file_path)
            return trn, val
        elif type == 'random':
            shuffled = self.shuffle()
            train_size = int(0.95 * len(shuffled))
            val_size = int(0.05 * len(shuffled))
            trn = shuffled[:train_size]
            val = shuffled[train_size:]
            torch.save([trn, val], file_path)
            return trn, val
        else:
            print('Unknown split mode !')

    def get_edges(self, n_nodes):
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                rows.append(i)
                cols.append(j)
        edges = [rows, cols]
        return edges



