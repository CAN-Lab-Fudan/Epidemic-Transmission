import h5py
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, RepeatedKFold
import sklearn.utils
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch
import torch.nn as nn
from torch import optim
from torch_geometric.loader import DataLoader
import scipy.sparse as sp
import matplotlib.pyplot as plt
import scienceplots
from tqdm import tqdm

from typing import Any, Tuple
import os
import os.path as osp
from parameters import parameter_parser
from utils import threshold, ER_Random_Graph
from trainer import Trainer
from layers_new import DynamicGAT

class Cross_Dataset(Dataset):
    def __init__(
        self,
        adj: np.array,
        data: np.ndarray,
        idx_array: np.array,
        dataset_type: bool = True,
    ) -> None:
        super(Cross_Dataset, self).__init__()
        self.all_data = data
        self.idx_array = idx_array
        self.dataset_type = dataset_type

        ed = sp.coo_matrix(adj)
        indices = np.vstack((ed.row, ed.col))
        self.edge_index = torch.tensor(data=indices, dtype=torch.long)

    def __len__(self):
        return self.idx_array.shape[0]
    
    def __getitem__(self, index) -> Any:
        target_index = self.idx_array[index]
        x = self.all_data[target_index, 1: -1]
        x = torch.tensor(data=x, dtype=torch.float).unsqueeze(dim=-1)
        num_infected = torch.sum(input=x)

        lambda_x = self.all_data[target_index, 0]
        lambda_x = torch.tensor(data=lambda_x, dtype=torch.float).unsqueeze(dim=-1)
        # y = self.all_data[target_index, -1].astype(float)
        # y = torch.tensor(data=y, dtype=torch.float).unsqueeze(dim=-1).unsqueeze(dim=-1)

        y = torch.tensor([[num_infected]], dtype=torch.float)

        if self.dataset_type: 
            # True 表示是训练集, 标签为y； False表示测试集，标签为lambda_x
            cur_data = Data(x=x, edge_index=self.edge_index, y=y)
        else:
            cur_data = Data(x=x, edge_index=self.edge_index, y=lambda_x)

        return cur_data

def fit_model(args, model, dataset, model_pth):
    print('\n Training started \n')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
    loss = nn.MSELoss()
    epochs = args.epochs

    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
        for i, data in enumerate(train_loader):
            data = data.to(device)
            prediction = model.forward(data)
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
    torch.save(model.state_dict(), model_pth)

def test_and_get_threshold(
        model,
        test_dataset,
        model_path: str,
    ):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    output = []
    test_loader = DataLoader(dataset=test_dataset)
    with torch.no_grad():
        for loader in test_loader:
            loader = loader.to(device)
            pre = model.forward(data=loader)
            y = loader.y
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

    return lambda_c_prime



if __name__ == "__main__":
    args = parameter_parser()
    N, avg_k = args.N, 4
    # G = ER_Random_Graph(N=N, avg_k=avg_k)
    group_name = f'Erdos_Renyi_Graph_N={N}'
    # group_name = f'Erdos_Renyi_Graph_N={N}_avg_k={avg_k}'
    # 读取所有的数据，将其放在一个名为all_data的array中，其中第0行表示对应的有效传播率lambda。
    with h5py.File(args.file_path + '.hdf5', 'r') as f:
        cur_group = f[group_name]
        adj = np.array(cur_group['adj'])

        all_data = cur_group['data']
        all_data = np.array(all_data[:, :])
        f.close()

    G = nx.from_numpy_array(A=adj)
    print(G.number_of_nodes())
    threshold_QMF, threshold_HMF = threshold(G=G)
    df = pd.DataFrame(index=all_data[:, 0], data=all_data[:, 1:-1])

    all_idx = set(df.index.values)
    mean = {}
    for idx in all_idx:
        cur_data = df.loc[idx].values
        data_mean = np.mean(np.sum(cur_data, axis=-1))
        mean[idx] = data_mean
    df_mean = pd.Series(data=mean).sort_index()

    # 对数据集进行k折交叉验证并使用保存最后的模型，模型保存的路径为 ./cross_validation/model_{idx}.pt
    model_folder = "./cross_validation"
    os.makedirs(model_folder, exist_ok=True)

    # kf = KFold(n_splits=10, shuffle=True, random_state=520)
    # 与正文8：2分割类似，采用5折验证，但是要进行10次实验。
    kf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=520)
    # 随机打乱all_data然后重复进行10折交叉验证
    rng = np.random.default_rng(520)
    rng.shuffle(x=all_data, axis=0)
    lambda_pre = []
    idx = 0
    for idx, (train_id, test_id) in tqdm(enumerate(kf.split(X=all_data[: ,1: -1]))):
        # print(all_data.shape, train_id.shape, test_id.shape, "\n")
        train_dataset = Cross_Dataset(adj=adj, data=all_data, idx_array=train_id, dataset_type=True)
        test_dataset = Cross_Dataset(adj=adj, data=all_data, idx_array=test_id, dataset_type=False)
        model = DynamicGAT(args=args)
        model_pth=osp.join(model_folder, "model_num_{}.pt".format(idx+1))
        fit_model(args=args, model=model, dataset=train_dataset, model_pth=model_pth)
        lambda_c_prime = test_and_get_threshold(model=model, test_dataset=test_dataset, model_path=model_pth)
        lambda_pre.append(lambda_c_prime)
        print(lambda_c_prime)
        # break


    # 作图，展示结果的变化
    np.savetxt(fname=osp.join(model_folder, "thresholds.txt"), X=np.array(lambda_pre))
    idx += 1
    mse = np.mean((np.array(lambda_pre) - threshold_HMF)**2)
    with plt.style.context(['science', 'ieee']):
        figure = plt.figure()
        # plt.title(r'The Fluctuation on ER Graph with (N={},$\bar k$={})'.format(args.N, avg_k))
        plt.plot(np.arange(idx)+1, lambda_pre, ls=":", marker='o', color='blue', label=r'$\lambda_c^\prime$')
        # plt.plot(x, avg_sqs, ls=':', color='green', label=r'$\lambda_c^{SQS}$')
        plt.axhline(y=threshold_HMF, ls='--', color='black', label=r'$\lambda_c^{HMF}$')
        plt.axhline(y=threshold_QMF, ls='-.', color='red', label=r'$\lambda_c^{QMF}$')

        plt.xlabel(r'$k$')
        plt.ylabel(r'$\lambda$')
        plt.ylim((0.0, 0.2))
        plt.text(0.8, 0.2, "MSE={:.4f}".format(mse), horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.legend()
        plt.savefig(osp.join(model_folder, "threshold.pdf"))

        plt.close()

