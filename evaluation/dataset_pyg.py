import os
import os.path as osp
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import h5py
from itertools import repeat, compress
from collections import defaultdict
from functools import lru_cache

import torch
from torch_geometric.data import Data, InMemoryDataset, download_url, Dataset
from torch_geometric.seed import seed_everything

from collections import defaultdict
try:
    from rdkit.Chem.Scaffolds import MurckoScaffold
except:
    MurckoScaffold = None
    print('Please install rdkit for data processing')
    
"""
ADMET Dataset
"""
class ADMET_Dataset(InMemoryDataset):

    def __init__(self, root, dataset=None, split_mode='random', split_seed=0, transform=None, pre_transform=None, pre_filter=None):
        assert split_mode in ['random', 'scaffold']

        self.root = root
        self.dataset = dataset
        self.split_seed = split_seed
        self.split_mode = split_mode

        super(ADMET_Dataset, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(f'{self.processed_dir}/{self.dataset}_processed.pt')
        trn, val, test = self.split(type = self.split_mode, seed = self.split_seed)
        self.train, self.val, self.test = trn, val, test

    @property
    def raw_dir(self):
        return f'{self.root}/raw/{self.dataset}'

    @property
    def processed_dir(self):
        return f'{self.root}/processed/{self.dataset}'

    @property
    def raw_file_names(self):
        name = [f'{self.dataset}_raw.hdf5']
        return name

    @property
    def processed_file_names(self):
        return [f'{self.dataset}_processed.pt']
    
    def process(self):
        """
        Process the raw data 2 processed data
        """
        data_list = []

        with h5py.File(f'{self.raw_dir}/{self.dataset}_raw.hdf5', 'r') as all_data_hdf5:

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
                if self.dataset == 'BACE':
                    data.label = torch.from_numpy(np.array(all_data_hdf5[f'{cid}']['Label'][()][:, 1:])).to(torch.int64)
                else:     
                    data.label = torch.from_numpy(np.array(all_data_hdf5[f'{cid}']['Label'][()])).to(torch.float32)
                data_list.append(data)

        print('making processed files:', self.processed_dir)
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), f'{self.processed_dir}/{self.dataset}_processed.pt')

    def split(self, type, seed):
        file_path = osp.join(f'{self.processed_dir}/{self.dataset}_{self.split_mode}_{self.split_seed}.pt')
        seed_everything(seed)
        if os.path.exists(file_path):
            trn, val, test = torch.load(file_path)
            return trn, val, test
        elif type == 'random':
            shuffled, split_perm = self.shuffle(True)
            train_size = int(0.8 * len(shuffled))
            val_size = int(0.1 * len(shuffled))
            trn = shuffled[:train_size]
            val = shuffled[train_size:(train_size + val_size)]
            test = shuffled[(train_size + val_size):]
            torch.save([trn, val, test], file_path)
            return trn, val, test
        elif type == 'scaffold':
            shuffled, split_perm = self.shuffle(True)
            trn, val, test = random_scaffold_split(dataset=shuffled, smiles_list=shuffled.data.smi, null_value=-1, seed=self.split_seed, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
            torch.save([trn, val, test], file_path)
            return trn, val, test
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

"""
Cliff Dataset
"""
class Cliff_Dataset(InMemoryDataset):

    def __init__(self, root, dataset=None, split_mode='train', transform=None, pre_transform=None, pre_filter=None):
        assert split_mode in ['train', 'test']

        self.root = root
        self.dataset = dataset
        self.split_mode = split_mode

        super(Cliff_Dataset, self).__init__(root, transform, pre_transform, pre_filter)

        if self.split_mode == 'train':
            self.data, self.slices = torch.load('{}/{}_processed.pt'.format(self.processed_dir, self.dataset))[self.split_mode]
        if self.split_mode == 'test':
            self.data, self.slices = torch.load('{}/{}_processed.pt'.format(self.processed_dir, self.dataset))[self.split_mode]
        self.data

    @property
    def raw_dir(self):
        return f'{self.root}/raw/{self.dataset}'

    @property
    def processed_dir(self):
        return f'{self.root}/processed/{self.dataset}'

    @property
    def raw_file_names(self):
        name = [f'{self.dataset}_raw.hdf5']
        return name

    @property
    def processed_file_names(self):
        return [f'{self.dataset}_processed.pt']
    
    def process(self):
        """
        Process the raw data 2 processed data
        """

        total_data_pd = pd.read_csv(f'{self.root}/raw/info/{self.dataset}.csv')
        train_data_pd = total_data_pd[total_data_pd['split'] == 'train']
        test_data_pd = total_data_pd[total_data_pd['split'] == 'test']
        all_molecular_list_train = list(train_data_pd['CID'])
        all_molecular_list_test = list(test_data_pd['CID'])
        
        data_list_dict = {}
        data_list_train = []
        data_list_test = []
            
        with h5py.File(f'{self.root}/raw/MoleculeACE_raw.hdf5', 'r') as all_data_hdf5:

            i = 0 
            for idx in tqdm(range(len(all_molecular_list_train)), desc = f'{i+1}/{len(all_molecular_list_train)}'):
                
                print('name', idx)
                
                i += 1
                cid = all_molecular_list_train[idx]
                node_num = int(all_data_hdf5[f'{cid}']['XYZ'].shape[0])
                data = Data()
                data.__num_nodes__ = node_num
                data.CID = cid
                # data.smi = all_data_hdf5[f'{cid}']['SMILES'][()].decode('utf-8')
                data.xyz = torch.from_numpy(all_data_hdf5[f'{cid}']['XYZ'][:]).to(torch.float32)
                data.global_MD = torch.from_numpy(np.where(np.isnan(all_data_hdf5[f'{cid}']['Global_MD_norm'][:]), 0, all_data_hdf5[f'{cid}']['Global_MD_norm'][:])).to(torch.float32).reshape(1, -1)
                data.global_FP = torch.from_numpy(np.where(np.isnan(all_data_hdf5[f'{cid}']['Global_FP'][:]), 0, all_data_hdf5[f'{cid}']['Global_FP'][:])).to(torch.float32).reshape(1, -1)
                data.node_features = torch.from_numpy(np.concatenate((all_data_hdf5[f'{cid}']['Node_NBO'][:], all_data_hdf5[f'{cid}']['Node_normal'][:]), axis = 1)).to(torch.float32)
                data.edge_index = torch.from_numpy(np.array(self.get_edges(node_num))).to(torch.int64)
                data.edge_features = torch.from_numpy(np.concatenate((all_data_hdf5[f'{cid}']['Edge_NBO'][:], all_data_hdf5[f'{cid}']['Edge_normal'][:]), axis = 1)).to(torch.float32)  
                data.label_y = torch.tensor(train_data_pd[train_data_pd['CID'] == cid]['y'].item()).to(torch.float32)
                data.label_y_pEC50pki = torch.tensor(train_data_pd[train_data_pd['CID'] == cid]['y [pEC50/pKi]'].item()).to(torch.float32)                
                data.cliff_mol = torch.tensor(train_data_pd[train_data_pd['CID'] == cid]['cliff_mol'].item()).to(torch.float32)
                data_list_train.append(data)
            data_list_dict['train'] = self.collate(data_list_train)

            i = 0 
            for idx in tqdm(range(len(all_molecular_list_test)), desc = f'{i+1}/{len(all_molecular_list_test)}'):
                i += 1
                cid = all_molecular_list_test[idx]
                node_num = int(all_data_hdf5[f'{cid}']['XYZ'].shape[0])
                data = Data()
                data.__num_nodes__ = node_num
                data.CID = cid
                # data.smi = all_data_hdf5[f'{cid}']['SMILES'][()].decode('utf-8')
                data.xyz = torch.from_numpy(all_data_hdf5[f'{cid}']['XYZ'][:]).to(torch.float32)
                data.global_MD = torch.from_numpy(np.where(np.isnan(all_data_hdf5[f'{cid}']['Global_MD_norm'][:]), 0, all_data_hdf5[f'{cid}']['Global_MD_norm'][:])).to(torch.float32).reshape(1, -1)
                data.global_FP = torch.from_numpy(np.where(np.isnan(all_data_hdf5[f'{cid}']['Global_FP'][:]), 0, all_data_hdf5[f'{cid}']['Global_FP'][:])).to(torch.float32).reshape(1, -1)
                data.node_features = torch.from_numpy(np.concatenate((all_data_hdf5[f'{cid}']['Node_NBO'][:], all_data_hdf5[f'{cid}']['Node_normal'][:]), axis = 1)).to(torch.float32)
                data.edge_index = torch.from_numpy(np.array(self.get_edges(node_num))).to(torch.int64)
                data.edge_features = torch.from_numpy(np.concatenate((all_data_hdf5[f'{cid}']['Edge_NBO'][:], all_data_hdf5[f'{cid}']['Edge_normal'][:]), axis = 1)).to(torch.float32)  
                data.label_y = torch.tensor(test_data_pd[test_data_pd['CID'] == cid]['y'].item()).to(torch.float32)
                data.label_y_pEC50pki = torch.tensor(test_data_pd[test_data_pd['CID'] == cid]['y [pEC50/pKi]'].item()).to(torch.float32)                
                data.cliff_mol = torch.tensor(test_data_pd[test_data_pd['CID'] == cid]['cliff_mol'].item()).to(torch.float32)
                data_list_test.append(data)
            data_list_dict['test'] = self.collate(data_list_train)
            
        print('making processed files:', self.processed_dir)
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(data_list_dict, f'{self.processed_dir}/{self.dataset}_processed.pt')
    def get_edges(self, n_nodes):
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                rows.append(i)
                cols.append(j)
        edges = [rows, cols]
        return edges

def random_scaffold_split(dataset, smiles_list, task_idx=None, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    if task_idx != None:
        y_task = np.array([data.label[:, task_idx].item() for data in dataset])
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    scaffolds = defaultdict(list)
    for ind, smiles in smiles_list:
        try:
            scaffold = generate_scaffold(smiles, include_chirality=True)
        except:
            print('This SMILES is Wrong:', smiles)
            continue
        scaffolds[scaffold].append(ind)
    scaffold_sets = list(scaffolds.values())
    np.random.shuffle(scaffold_sets)
    n_total_valid = int(np.floor(frac_valid * len(dataset)))
    n_total_test = int(np.floor(frac_test * len(dataset)))
    train_idx = []
    valid_idx = []
    test_idx = []
    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)
    # print(f'Split dataset into {len(train_idx)} train, {len(valid_idx)} valid, {len(test_idx)} test')
    # print(train_idx)
    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    return train_dataset, valid_dataset, test_dataset

def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold

"""
PLI Opioids Dataset
"""
class PLI_Opioids_Dataset(InMemoryDataset):
    def __init__(self, root, dataset=None, split_mode='random', split_seed=0, transform=None, pre_transform=None, pre_filter=None):
        assert dataset in ['CYP2D6_reg', 'CYP3A4_reg', 'DOR_reg', 'KOR_reg', 'MDR1_reg', 'MOR_reg']
        assert split_mode in ['random', 'scaffold']

        self.root = root
        self.dataset = dataset
        self.split_seed = split_seed
        self.split_mode = split_mode

        super(PLI_Opioids_Dataset, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(f'{self.processed_dir}/{self.dataset}_processed.pt')
        trn, val, test = self.split(type = self.split_mode, seed = self.split_seed)
        self.train, self.val, self.test = trn, val, test

    @property
    def raw_dir(self):
        return f'{self.root}/raw/{self.dataset}'

    @property
    def processed_dir(self):
        return f'{self.root}/processed/PLI/Opioids{self.dataset}'

    @property
    def raw_file_names(self):
        name = [f'{self.dataset}_raw.hdf5']
        return name

    @property
    def processed_file_names(self):
        return [f'{self.dataset}_processed.pt']
    
    def process(self):
        """
        Process the raw data 2 processed data
        """
        data_list = []
        task_data_pd = pd.read_csv(f'{self.root}/raw/info/{self.dataset}.csv')
        all_molecular_list = task_data_pd['CID']

        # with h5py.File(f'{self.raw_dir}/Opioids_raw.hdf5', 'r') as all_data_hdf5:
        with h5py.File(f'/export/disk7/why/dataset/Opioids/h5_file/Opioids_raw.hdf5', 'r') as all_data_hdf5:

            i = 0 
            for idx in tqdm(range(len(all_molecular_list)), desc = f'{i+1}/{len(all_molecular_list)}'):
                i += 1
                
                cid = all_molecular_list[idx]
                node_num = int(all_data_hdf5[f'{cid}']['XYZ'].shape[0])
                data = Data()
                data.__num_nodes__ = node_num
                data.CID = cid
                data.smi = task_data_pd[task_data_pd['CID'] == cid]['SMILES'].values[0]
                data.xyz = torch.from_numpy(all_data_hdf5[f'{cid}']['XYZ'][:]).to(torch.float32)
                data.global_MD = torch.from_numpy(np.where(np.isnan(all_data_hdf5[f'{cid}']['Global_MD_norm'][:]), 0, all_data_hdf5[f'{cid}']['Global_MD_norm'][:])).to(torch.float32).reshape(1, -1)
                data.global_FP = torch.from_numpy(np.where(np.isnan(all_data_hdf5[f'{cid}']['Global_FP'][:]), 0, all_data_hdf5[f'{cid}']['Global_FP'][:])).to(torch.float32).reshape(1, -1)
                data.node_features = torch.from_numpy(np.concatenate((all_data_hdf5[f'{cid}']['Node_NBO'][:], all_data_hdf5[f'{cid}']['Node_normal'][:]), axis = 1)).to(torch.float32)
                data.edge_index = torch.from_numpy(np.array(self.get_edges(node_num))).to(torch.int64)
                data.edge_features = torch.from_numpy(np.concatenate((all_data_hdf5[f'{cid}']['Edge_NBO'][:], all_data_hdf5[f'{cid}']['Edge_normal'][:]), axis = 1)).to(torch.float32)  
                data.label = torch.from_numpy(task_data_pd[task_data_pd['CID'] == cid]['label'].values).to(torch.float32)
                data_list.append(data)

        print('making processed files:', self.processed_dir)
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), f'{self.processed_dir}/{self.dataset}_processed.pt')

    def split(self, type, seed):
        file_path = osp.join(f'{self.processed_dir}/{self.dataset}_{self.split_mode}_{self.split_seed}.pt')
        seed_everything(seed)
        if os.path.exists(file_path):
            trn, val, test = torch.load(file_path)
            return trn, val, test
        elif type == 'random':
            shuffled, split_perm = self.shuffle(True)
            train_size = int(0.8 * len(shuffled))
            val_size = int(0.1 * len(shuffled))
            trn = shuffled[:train_size]
            val = shuffled[train_size:(train_size + val_size)]
            test = shuffled[(train_size + val_size):]
            torch.save([trn, val, test], file_path)
            return trn, val, test
        elif type == 'scaffold':
            shuffled, split_perm = self.shuffle(True)
            trn, val, test = random_scaffold_split(dataset=shuffled, smiles_list=shuffled.data.smi, null_value=-1, seed=self.split_seed, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
            torch.save([trn, val, test], file_path)
            return trn, val, test
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

"""
PLI Common Dataset
"""
class PLI_Common_Dataset_protein(Dataset):
    def __init__(self, root, info_path, dataset=None, name='protein', split_mode='asPaper', split_seed=0, mode='train', transform=None, pre_transform=None, pre_filter=None):
        assert dataset in ['BindingDB', 'Celegans', 'Human', 'KIBA', 'DAVIS']
        self.root = root
        self.info_path = info_path
        self.dataset = dataset
        self.name = name
        self.split_mode = split_mode
        self.split_seed = split_seed
        self.mode=mode

        self.data_pd = pd.read_csv(f'{self.root}/raw/info/{self.info_path}')
        self.data_pd['PLI_ID'] = self.data_pd['CID'] + '_AND_' + self.data_pd['ProteinName']
        self.train_data_pd = self.data_pd[self.data_pd['Mode'] == 'Train'][['SMILES','CID', 'SEQ', 'ProteinName', 'Label', 'PLI_ID', 'Mode']]
        self.valid_data_pd = self.data_pd[self.data_pd['Mode'] == 'Valid'][['SMILES','CID', 'SEQ', 'ProteinName', 'Label', 'PLI_ID', 'Mode']]
        self.test_data_pd = self.data_pd[self.data_pd['Mode'] == 'Test'][['SMILES','CID', 'SEQ', 'ProteinName', 'Label', 'PLI_ID', 'Mode']]

        super(PLI_Common_Dataset_protein, self).__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_dir(self):
        return f'{self.root}/raw/{self.dataset}/{self.name}'
    @property
    def processed_dir(self):
        return f'{self.root}/processed/{self.dataset}/{self.name}'
    @property
    def raw_file_names(self):
        name = []
        return name
    @property
    def processed_file_names(self):
        all_names = set(list(self.train_data_pd['ProteinName']) + list(self.valid_data_pd['ProteinName']) + list(self.test_data_pd['ProteinName']))
        return ['{}.pt'.format(name) for name in all_names]
    
    def process(self):
        data_pd_dict = {
            'train':self.train_data_pd,
            'valid':self.valid_data_pd,
            'test':self.test_data_pd
        }

        with h5py.File(f'{self.raw_dir}/{self.dataset}_raw.h5py') as f:
            for data_key in data_pd_dict:
                print('Processing the {} set'.format(data_key))
                data_pd = data_pd_dict[data_key]
                for i in range(len(data_pd)):
                    data = Data()
                    name = data_pd.iloc[i]['ProteinName']
                    data.protein_name = name
                    data.gloabl_protein = data_pd.iloc[i]['SEQ']
                    data.global_protein_emb = torch.from_numpy(f[name]['embedding'][:])
                    torch.save(data, '{}/{}.pt'.format(self.processed_dir, data_pd.iloc[i]['ProteinName']))

    def len(self):
        if self.mode=='train':
            return len(self.train_data_pd)
        elif self.mode=='valid':
            return len(self.valid_data_pd)
        elif self.mode=='test':
            return len(self.test_data_pd)
        else:
            print('Error!')
            pass

    @lru_cache(maxsize=15000)
    def get(self, idx):
        if self.mode=='train':
            data = torch.load('{}/{}.pt'.format(self.processed_dir, list(self.train_data_pd['ProteinName'])[idx]))
        elif self.mode=='valid':
            data = torch.load('{}/{}.pt'.format(self.processed_dir, list(self.valid_data_pd['ProteinName'])[idx]))
        elif self.mode=='test':
            data = torch.load('{}/{}.pt'.format(self.processed_dir, list(self.test_data_pd['ProteinName'])[idx]))
        else:
            print('Error')
        return data


class PLI_Common_Dataset_ligand(Dataset):
    def __init__(self, root, info_path, dataset=None, name='ligand', split_mode='asPaper', split_seed=0, mode='train', transform=None, pre_transform=None, pre_filter=None):
        assert dataset in ['BindingDB', 'Celegans', 'Human', 'KIBA', 'DAVIS']
        self.root = root
        self.info_path = info_path
        self.dataset = dataset
        self.name = name
        self.split_mode = split_mode
        self.split_seed = split_seed
        self.mode=mode
        
        self.data_pd = pd.read_csv(f'{self.root}/raw/info/{self.info_path}')
        self.data_pd['PLI_ID'] = self.data_pd['CID'] + '_AND_' + self.data_pd['ProteinName']
        self.train_data_pd = self.data_pd[self.data_pd['Mode'] == 'Train'][['SMILES','CID', 'SEQ', 'ProteinName', 'Label', 'PLI_ID', 'Mode']]
        self.valid_data_pd = self.data_pd[self.data_pd['Mode'] == 'Valid'][['SMILES','CID', 'SEQ', 'ProteinName', 'Label', 'PLI_ID', 'Mode']]
        self.test_data_pd = self.data_pd[self.data_pd['Mode'] == 'Test'][['SMILES','CID', 'SEQ', 'ProteinName', 'Label', 'PLI_ID', 'Mode']]

        super(PLI_Common_Dataset_ligand, self).__init__(root, transform, pre_transform, pre_filter)


    @property
    def raw_dir(self):
        return f'{self.root}/raw/{self.dataset}/{self.name}'
    @property
    def processed_dir(self):
        return f'{self.root}/processed/{self.dataset}/{self.name}'
    @property
    def raw_file_names(self):
        name = []
        return name
    @property
    def processed_file_names(self):
        all_names = set(list(self.train_data_pd['CID']) + list(self.valid_data_pd['CID']) + list(self.test_data_pd['CID']))
        return ['{}.pt'.format(name) for name in all_names]

    def process(self):

        data_pd_dict = {
            'train':self.train_data_pd,
            'valid':self.valid_data_pd,
            'test':self.test_data_pd
        }

        with h5py.File(f'{self.raw_dir}/{self.dataset}_raw.hdf5', 'r') as all_data_hdf5:

            for data_key in data_pd_dict.keys():
                print('Processing the {} set'.format(data_key))
                data_pd = data_pd_dict[data_key]            
                for i in range(len(data_pd)):
                    cid = data_pd.iloc[i]['CID']

                    node_num = int(all_data_hdf5[f'{cid}']['XYZ'].shape[0])
                    data = Data()
                    data.__num_nodes__ = node_num                    
                    data.ligand_name = cid
                    data.ligand_smi = data_pd.iloc[i]['SMILES']
                    data.xyz = torch.from_numpy(all_data_hdf5[f'{cid}']['XYZ'][:]).to(torch.float32)
                    data.global_MD = torch.from_numpy(np.where(np.isnan(all_data_hdf5[f'{cid}']['Global_MD_norm'][:]), 0, all_data_hdf5[f'{cid}']['Global_MD_norm'][:])).to(torch.float32).reshape(1, -1)
                    data.global_FP = torch.from_numpy(np.where(np.isnan(all_data_hdf5[f'{cid}']['Global_FP'][:]), 0, all_data_hdf5[f'{cid}']['Global_FP'][:])).to(torch.float32).reshape(1, -1)
                    data.node_features = torch.from_numpy(np.concatenate((all_data_hdf5[f'{cid}']['Node_NBO'][:], all_data_hdf5[f'{cid}']['Node_normal'][:]), axis = 1)).to(torch.float32)
                    data.edge_index = torch.from_numpy(np.array(self.get_edges(node_num))).to(torch.int64)
                    data.edge_features = torch.from_numpy(np.concatenate((all_data_hdf5[f'{cid}']['Edge_NBO'][:], all_data_hdf5[f'{cid}']['Edge_normal'][:]), axis = 1)).to(torch.float32)  
                    torch.save(data, '{}/{}.pt'.format(self.processed_dir, data_pd.iloc[i]['QMID']))

    def len(self):
        if self.mode=='train':
            return len(self.train_data_pd)
        elif self.mode=='valid':
            return len(self.valid_data_pd)
        elif self.mode=='test':
            return len(self.test_data_pd)
        else:
            print('Error!')
            pass

    @lru_cache(maxsize=50000)
    def get(self, idx):
        if self.mode=='train':
            data = torch.load('{}/{}.pt'.format(self.processed_dir, self.train_data_pd.iloc[idx]['CID']))
        elif self.mode=='valid':
            data = torch.load('{}/{}.pt'.format(self.processed_dir, self.valid_data_pd.iloc[idx]['CID']))
        elif self.mode=='test':
            data = torch.load('{}/{}.pt'.format(self.processed_dir, self.test_data_pd.iloc[idx]['CID']))
        else:
            print('Error')
        return data
        
    def get_edges(self, n_nodes):
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                rows.append(i)
                cols.append(j)
        edges = [rows, cols]
        return edges

class PLI_Common_Dataset_label(Dataset):
    def __init__(self, root, dataset=None, name='label', split_mode='asPaper', split_seed=0, mode='train', transform=None, pre_transform=None, pre_filter=None):
        assert dataset in ['BindingDB', 'Celegans', 'Human', 'KIBA', 'DAVIS']
        self.root = root
        self.info_path = info_path
        self.dataset = dataset
        self.name = name
        self.split_mode = split_mode
        self.split_seed = split_seed
        self.mode=mode
        
        self.data_pd = pd.read_csv(f'{self.root}/raw/info/{self.info_path}')
        self.data_pd['PLI_ID'] = self.data_pd['CID'] + '_AND_' + self.data_pd['ProteinName']
        self.train_data_pd = self.data_pd[self.data_pd['Mode'] == 'Train'][['SMILES','CID', 'SEQ', 'ProteinName', 'Label', 'PLI_ID', 'Mode']]
        self.valid_data_pd = self.data_pd[self.data_pd['Mode'] == 'Valid'][['SMILES','CID', 'SEQ', 'ProteinName', 'Label', 'PLI_ID', 'Mode']]
        self.test_data_pd = self.data_pd[self.data_pd['Mode'] == 'Test'][['SMILES','CID', 'SEQ', 'ProteinName', 'Label', 'PLI_ID', 'Mode']]

        super(PLI_Common_Dataset_ligand, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_dir(self):
        return f'{self.root}/raw/{self.dataset}/{self.name}'
    @property
    def processed_dir(self):
        return f'{self.root}/processed/{self.dataset}/{self.name}'
    @property
    def raw_file_names(self):
        name = []
        return name
    @property
    def processed_file_names(self):
        all_names = set(list(self.train_data_pd['PLI_ID'] + '_' + self.train_data_pd['Mode']) + list(self.valid_data_pd['PLI_ID'] + '_' + self.valid_data_pd['Mode']) + list(self.test_data_pd['PLI_ID'] + '_' + self.test_data_pd['Mode']))
        return ['label_{}.pt'.format(name) for name in all_names]
    
    def process(self):
        data_pd_dict = {
            'train':self.train_data_pd,
            'valid':self.valid_data_pd,
            'test':self.test_data_pd
        }
        for data_key in data_pd_dict.keys():
            print('Processing the {} set'.format(data_key))
            data_pd = data_pd_dict[data_key]
            for i in range(len(data_pd)):
                data = Data()
                data.label = torch.tensor(data_pd.iloc[i]['Label']).to(torch.float32)
                torch.save(data, '{}/label_{}_{}.pt'.format(self.processed_dir, data_pd.iloc[i]['PLI_ID'], data_pd.iloc[i]['Mode']))

    def len(self):
        if self.mode=='train':
            return len(self.train_data_pd)
        elif self.mode=='valid':
            return len(self.valid_data_pd)
        elif self.mode=='test':
            return len(self.test_data_pd)
        else:
            print('Error!')
            pass

    @lru_cache(maxsize=60000)
    def get(self, idx):
        if self.mode=='train':
            data = torch.load('{}/label_{}_{}.pt'.format(self.processed_dir, self.train_data_pd.iloc[idx]['PLI_ID'], self.train_data_pd.iloc[idx]['Mode']))
        elif self.mode=='valid':
            data = torch.load('{}/label_{}_{}.pt'.format(self.processed_dir, self.valid_data_pd.iloc[idx]['PLI_ID'], self.valid_data_pd.iloc[idx]['Mode']))
        elif self.mode=='test':
            data = torch.load('{}/label_{}_{}.pt'.format(self.processed_dir, self.test_data_pd.iloc[idx]['PLI_ID'], self.test_data_pd.iloc[idx]['Mode']))
        else:
            print('Error')
        return data

class TripleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_protein, dataset_ligand, dataset_label):
        self.dataset_protein = dataset_protein
        self.dataset_ligand = dataset_ligand
        self.dataset_label = dataset_label
        assert dataset_protein.mode == dataset_ligand.mode == dataset_label.mode
        self.mode = dataset_protein.mode
        assert dataset_protein.len() == dataset_ligand.len() == dataset_label.len()
        self.len = dataset_protein.len()

    def __getitem__(self, idx):
        return self.dataset_protein[idx], self.dataset_ligand[idx], self.dataset_label[idx]
    
    def __len__(self):
        return self.len

from torch_geometric.data import Batch
def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    batchC = Batch.from_data_list([data[2] for data in data_list])
    return batchA, batchB, batchC
