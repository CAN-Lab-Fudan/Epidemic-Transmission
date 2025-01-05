import networkx as nx
import glob
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch_geometric.typing import Adj
from typing import Tuple, Any
from torch_geometric.loader import DataLoader

from layers_new import DynamicGAT
from SQS_simulation import SQS_simulation, DynamicsSIS
from dataset import MyDataset


class Trainer:

    def __init__(self, args, group_name: str):
        self.model = None
        self.group_name = group_name
        self.args = args
        self.set_model()

        self.root = os.path.join(args.file_path, group_name)
        self.model_path = os.path.join(self.root, self.args.model_path)

    def set_model(self):
        self.model = DynamicGAT(args=self.args)

    def run_data(self, G, lambda_range):
        SQS = DynamicsSIS(
            Graph=G,
            beta=lambda_range,
            file_path=self.args.file_path,
            group_name=self.group_name,
            numbers_every_miu=500,
        )
        SQS.multiprocess_simulation()

    def create_dataset(self):
        My_Dataset = MyDataset(
            file_path=self.args.file_path,
            dataset_path=self.group_name,
        )

        return My_Dataset

    def fit_model(self):
        print('\n Training started \n')

        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.model.train()

        # optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=0.001)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=0.001)
        loss = nn.MSELoss()
        epochs = self.args.epochs

        self.model = self.model.to(self.device)
        dataset = self.create_dataset()

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            train_loader = DataLoader(dataset=dataset, batch_size=self.args.batch_size, shuffle=True)
            for i, data in enumerate(train_loader):
                data = data.to(self.device)
                prediction = self.model.forward(data)
                target = data.y
                cur_loss = loss(prediction, target)

                optimizer.zero_grad()
                cur_loss.backward()
                optimizer.step()

                epoch_loss += cur_loss.item()

            epoch_loss = epoch_loss / (i + 1)

            print(
                f'After {epoch + 1} / {epochs} epochs, the average loss is: {epoch_loss} \n'
            )

            if (epoch + 1) % self.args.check_point == 0:
                print(f'When epoch={epoch + 1}, average loss on val data is:')
                self.val_model()


        torch.save(self.model.state_dict(), self.model_path)

        lambda_c_prime, df, df_var = self.test_and_get_threshold(model_path=self.model_path, group_name=self.group_name)

        print(f'After train, the threshold predicted by model is: {lambda_c_prime}\n')

        return lambda_c_prime, df, df_var

    def val_model(self):
        self.model.eval()
        loss = nn.L1Loss()
        all_loss = 0
        correct = 0
        data_list = glob.glob(os.path.join(
            self.root,
            'val_data',
            '*.pt'
        ))
        idx = 0
        for idx in range(len(data_list)):
            path = data_list[idx]
            loader = torch.load(path, map_location=self.device)
            loader = loader.to(self.device)
            pre = self.model.forward(loader)
            target = loader.y
            cur_loss = loss(pre, target)
            all_loss += cur_loss.item()
        avg_loss = all_loss / len(data_list)
        print(f'{avg_loss}\n')

    def test_and_get_threshold(
            self,
            model_path: str,
            group_name: str,
    ) -> Tuple[float, pd.Series]:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.model.to(device)
        output = []
        error= []
        test_path = glob.glob(os.path.join(self.args.file_path, group_name, 'test_data', '*.pt'))
        with torch.no_grad():
            for path in test_path:
                test_data = torch.load(path)
                test_data = test_data.to(device)
                pre = self.model.forward(data=test_data)
                y = test_data.y
                # y = y.cpu().numpy().item()
                y = y.cpu().numpy()
                pre = pre.cpu().numpy().item()
                cur_out = np.insert(arr=y, obj=1, values=pre)
                output.append(cur_out)

        output = np.array(output)
        df_out = pd.DataFrame(index=output[:, 0], data=output[:, 1:]).sort_index()

        out_mean = {}
        out_var = {}
        for idx in set(df_out.index.values):
            cur_df = df_out.loc[idx]
            # cur_error = df_error.loc[idx]
            cur_values = cur_df.iloc[:, 0].values
            real_values = cur_df.iloc[:, -1].values
            out_mean[idx] = np.mean(cur_values)
            if np.mean(cur_values) == 0:
                out_var[idx] = 0
            else:
                # values = cur_values - real_values
                out_var[idx] = np.std(cur_values) / np.mean(cur_values)
                # out_var[idx] = np.std(values) / np.mean(values)
        df_mean = pd.Series(out_mean).sort_index()
        df_var = pd.Series(out_var).sort_index()

        lambda_c_prime = df_var.idxmax()

        return lambda_c_prime, df_mean, df_var


