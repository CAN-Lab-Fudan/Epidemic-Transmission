import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch.utils.data import Dataset
import os
import h5py
from typing import Union, List, Tuple, Optional
import glob
from tqdm import tqdm
import shutil


class MyDataset(Dataset):

    def __init__(
            self,
            file_path: str,
            dataset_path: str,
            train_ratio: float = 0.7,
            val_ratio: float = 0.2,
    ):
        super(MyDataset, self).__init__()
        
        if not os.path.exists(file_path):
            os.mkdir(file_path)

        root = os.path.join(file_path, dataset_path)
        if not os.path.exists(root):
            os.mkdir(root)

        self.root = root
        self.dataset_path = dataset_path
        self.file_path = file_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.train_file = os.path.join(root, 'train_data')
        self.val_file = os.path.join(root, 'val_data')
        self.test_file = os.path.join(root, 'test_data')
        try:
            shutil.rmtree(self.train_file)
            os.mkdir(self.train_file)
        except OSError as e:
            os.mkdir(self.train_file)
        
        try:
            shutil.rmtree(self.val_file)
            os.mkdir(self.val_file)
        except OSError as e:
            os.mkdir(self.val_file)

        try:
            shutil.rmtree(self.test_file)
            os.mkdir(self.test_file)
        except OSError as e:
            os.mkdir(self.test_file)

        self.save_data_in_file()

    def __len__(self):
        return self.train_length

    def __getitem__(self, item):
        cur_data = torch.load(os.path.join(self.train_file, f'data_{item}.pt'))

        return cur_data

    def save_data_in_file(self):

        with h5py.File(self.file_path + '.hdf5', 'r') as f:
            cur_group = f[self.dataset_path]

            adj = np.array(cur_group['adj'])

            # adj = np.pad(array=adj, pad_width=((1,0), (1, 0)), constant_values=((1, 0), (1, 0)))
            # adj[0, 0] = 0

            ed = sp.coo_matrix(adj)
            indices = np.vstack((ed.row, ed.col))
            edge_index = torch.tensor(data=indices, dtype=torch.long)

            degrees = degree(index=edge_index[0], num_nodes=adj.shape[0], dtype=torch.long)

            all_data = cur_group['data']

            # 打乱顺序
            rng = np.random.default_rng()
            all_data = rng.permutation(x=all_data, axis=0)

            length = all_data.shape[0]

            self.train_length = int(self.train_ratio * length)
            self.val_length = int(self.val_ratio * length)

            train_idx = np.random.choice(a=length, size=self.train_length + self.val_length, replace=False)

            idx_train = 0
            idx_val = 0
            idx_test = 0

            for idx in range(length):
                cur_data = all_data[idx, 1: -1]
                num_nodes = len(cur_data)
                cur_label = all_data[idx, -1].astype(np.int)
                cur_lambda = all_data[idx, 0]
                # cur_lambda = torch.tensor(data=cur_lambda, dtype=torch.float).unsqueeze(dim=-1)

                data = torch.tensor(data=cur_data, dtype=torch.float32)

                num_infected = torch.sum(input=data)
                # data = torch.mul(data, degrees)
                # num_infected = num_infected
                # cur_lambda = torch.tensor(data=cur_lambda, dtype=torch.float).unsqueeze(dim=-1)
                # data = torch.cat((cur_lambda, data), dim=0)
                # num_nodes = num_nodes + 1
                y = torch.tensor([[num_infected]], dtype=torch.float)
                data = data.unsqueeze(dim=-1)

                # data = data * cur_lambda


                label = torch.tensor(data=cur_label, dtype=torch.long).unsqueeze(dim=-1)

                cur_data = Data(x=data, edge_index=edge_index, y=y, num_nodes=num_nodes)

                if idx in train_idx and idx_train < self.train_length:
                    torch.save(cur_data, os.path.join(self.train_file, f'data_{idx_train}.pt'))
                    idx_train += 1
                elif idx in train_idx and idx_val < self.val_length:
                    torch.save(cur_data, os.path.join(self.val_file, f'data_{idx_val}.pt'))
                    idx_val += 1
                else:
                    cur_y = torch.tensor([cur_lambda, num_infected], dtype=torch.float)
                    cur_data = Data(x=data, edge_index=edge_index, y=cur_y, num_nodes=num_nodes)
                    torch.save(cur_data, os.path.join(self.test_file, f'data_{idx_test}.pt'))
                    idx_test += 1


class Just_Test:
    def __init__(
            self,
            file_path: str,
            dataset_path: str,
    ):
        
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        self.file_path = file_path
        self.dataset_path = dataset_path

        root = os.path.join(file_path, dataset_path)

        if not os.path.exists(root):
            os.mkdir(root)

        self.root = root
        self.test_file = os.path.join(root, 'test_data')

        try:
            shutil.rmtree(self.test_file)
            os.mkdir(self.test_file)

        except OSError as e:
            os.mkdir(self.test_file)

        self.just_test()

    def just_test(self):
        with h5py.File(self.file_path + '.hdf5', 'r') as f:
            cur_group = f[self.dataset_path]

            adj = np.array(cur_group['adj'])

            # adj = np.pad(array=adj, pad_width=((1,0), (1, 0)), constant_values=((1, 0), (1, 0)))
            # adj[0, 0] = 0

            ed = sp.coo_matrix(adj)
            indices = np.vstack((ed.row, ed.col))
            edge_index = torch.tensor(data=indices, dtype=torch.long)

            degrees = degree(index=edge_index[0], num_nodes=adj.shape[0], dtype=torch.long)

            all_data = cur_group['data']

            # 打乱顺序
            rng = np.random.default_rng()
            all_data = rng.permutation(x=all_data, axis=0)

            length = all_data.shape[0]

            for idx in range(length):
                data = all_data[idx, 1: -1]
                num_nodes = len(data)
                label = all_data[idx, -1].astype(np.int)
                cur_lambda = all_data[idx, 0]
                cur_lambda = torch.tensor(data=cur_lambda, dtype=torch.float).unsqueeze(dim=-1)

                data = torch.tensor(data=data, dtype=torch.float32)
                # data = torch.mul(data, degrees)
                num_infected = torch.sum(input=data)

                # data = torch.cat((cur_lambda, data), dim=0)
                # num_nodes = num_nodes + 1

                data = data.unsqueeze(dim=-1)

                # data = data * cur_lambda

                label = torch.tensor(data=label, dtype=torch.long)
                cur_y = torch.tensor([cur_lambda, num_infected], dtype=torch.float)
                cur_data = Data(x=data, edge_index=edge_index, y=cur_y, num_nodes=num_nodes)
                # cur_data = Data(x=data, edge_index=edge_index, y=cur_lambda, num_nodes=num_nodes)
                torch.save(cur_data, os.path.join(self.test_file, f'data_{idx}.pt'))
