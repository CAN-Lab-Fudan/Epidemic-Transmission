import random

import networkx as nx
import numpy as np
import os
import csv
import h5py
import gzip
import shutil
from typing import Tuple, Optional, List

import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from typing import Union

from utils import threshold


class SQS_simulation:
    r"""
    Args:
        Graph: type of networkx.Graph where to run dynamics,
        lambda_range: change range of lambda,
        lambda_numbers: numbers the lambda change between the range,
        file_group: the file type hdf5's group,
        avg_numbers: the repeats numbers for each lambda,
        M: the numbers of the history state the dynamic have ever get,
        init_ratio: the rate infected nodes to initinize the network,
        steps: the numbers of snapshots for each lambda,
    """

    def __init__(
            self,
            Graph: nx.Graph,
            lambda_range: Tuple[float, float],
            lambda_numbers: int,
            file_path: str,
            group_name: str,
            avg_numbers: int = 50,
            M: int = 100,
            p_r: float = 0.02,
            init_ratio: Optional[float] = None,
            steps: int = 700,
            update_t: int = 200,
            dt: int = 5,
    ) -> None:

        self.Graph = Graph
        self.lambda_rannge = lambda_range
        self.lambda_numbers = lambda_numbers
        self.file_path = file_path

        self.group_name = group_name

        self.avg_numbers = avg_numbers
        self.M = M
        self.p_r = p_r
        self.init_ratio = init_ratio
        self.steps = steps
        self.update_t = update_t
        self.dt = dt

        self.numbers_of_node = nx.number_of_nodes(G=Graph)
        self.degrees = np.array(nx.degree(G=Graph))[:, -1]

        adj = nx.to_numpy_array(G=Graph)

        if not os.path.exists(file_path):
            os.mkdir(file_path)
        root = os.path.join(file_path, group_name)
        if not os.path.exists(root):
            os.mkdir(root)

        df_adj = pd.DataFrame(data=adj)
        df_adj.to_hdf(path_or_buf=os.path.join(root, 'data.hdf5'), key='adjust_metrix', index=False)

        self.root = root
        self.df = pd.DataFrame(data=None, columns=np.arange(self.numbers_of_node + 2))



    def init_infected_state(self) -> np.array:

        if not self.init_ratio:
            infected_set = np.random.choice(a=self.numbers_of_node, size=1, replace=False)
        else:
            init_infected_numbers = int(self.numbers_of_node * self.init_ratio)
            infected_set = np.random.choice(a=self.numbers_of_node, size=init_infected_numbers, replace=False)

        return infected_set

    def update_history_list(
            self,
            infected_set: np.array,
            history_list: List[np.array],
            update_rate: float,
    ) -> None:
        if len(history_list) < self.M:
            history_list.append(infected_set)
        else:
            p = np.random.random()
            if p <= update_rate:
                seed = np.random.choice(a=self.M, size=1, replace=False).item()
                history_list[seed] = infected_set

    def init_from_history(
            self,
            history_list: List[np.array],
    ) -> np.array:
        seed = np.random.choice(a=len(history_list), size=1, replace=False).item()
        infected_set = history_list[seed]

        return infected_set


    def run_get_one_vector(self, cur_lambda: float):
        df = pd.DataFrame(data=None, columns=np.arange(self.numbers_of_node + 2))
        history_list = []
        infected_set = self.init_infected_state()
        history_list.append(infected_set)

        lambda_c_QMF, lambda_c_HMF = threshold(G=self.Graph)

        label = 0
        if cur_lambda > lambda_c_HMF:
            label = 1
        else:
            label = np.random.choice(a=1, size=1, replace=False).item()

        step = 0
        while step < self.steps:

            # 进行传染病传播的模拟
            n_i = len(infected_set)
            n_k = np.sum(self.degrees[infected_set])

            recovery = np.divide(n_i, n_i + cur_lambda * n_k)
            dt = np.divide(1, n_i + cur_lambda * n_k)

            # 更新历史经历过的状态
            update_rate = np.multiply(self.p_r, dt)
            if np.random.random() <= update_rate:
                self.update_history_list(infected_set=infected_set, history_list=history_list, update_rate=update_rate)

            seed = np.random.random()
            if seed <= recovery:
                recovery_axis = np.random.choice(a=n_i, size=1, replace=False)
                infected_set = np.delete(arr=infected_set, obj=recovery_axis)

            else:
                chosen_ratio = np.divide(self.degrees[infected_set], n_k)
                cur_p = np.random.random()
                i = 1
                while np.sum(chosen_ratio[:i]) < cur_p:
                    i += 1
                chosen_node = infected_set[i - 1]
                neighbors = np.array(self.Graph[chosen_node])
                infecting_node = np.random.choice(a=neighbors, size=1)
                if infecting_node not in infected_set:
                    infected_set = np.append(arr=infected_set, values=infecting_node)

            if len(infected_set) == 0:
                infected_set = self.init_from_history(history_list=history_list)

            if step >= self.update_t and state % self.dt == 0:
                # 保存所有经历过的状态
                state = np.zeros(shape=self.numbers_of_node)
                state[infected_set] = 1
                cur_data = np.insert(arr=state, obj=0, values=cur_lambda)
                cur_data = np.append(arr=cur_data, values=label)

                cur_df = pd.DataFrame(data=np.array([cur_data]), columns=np.arange(self.numbers_of_node + 2))
                # df = df.append(other=cur_df, ignore_index=True)
                df = pd.concat((df, cur_df), axis=0, ignore_index=True)
            step += 1
        return df

    def multi_process(self):
        # 开始产生数据
        Lambda = np.linspace(start=self.lambda_rannge[0], stop=self.lambda_rannge[-1], num=self.lambda_numbers)

        pool = Pool(5)
        for cur_lambda in Lambda:
            pool.apply_async(func=self.run_get_one_vector, args=(cur_lambda,), callback=self.append_df)
        pool.close()
        pool.join()

        self.df.to_hdf(os.path.join(self.root, 'data.hdf5'), key='data', index=False)

    def append_df(self, df: pd.DataFrame):
        # self.df = self.df.append(other=df, ignore_index=True)
        self.df = pd.concat((self.df, df), axis=0, ignore_index=True)


class DynamicsSIS:

    def __init__(
        self,
        Graph: nx.Graph,
        beta: Tuple[float, float],
        file_path: str,
        group_name: str,
        ratio: Optional[float] = None,
        gama: float = 1.0,
        steps_to_steady: int = 50,
        numbers_bata_change: int = 100,
        numbers_every_miu: int = 300,
    ):
        self.Graph = Graph
        self.ratio = ratio
        self.steps_to_steady = steps_to_steady
        self.file_path = file_path
        self.group_name = group_name
        self.beta = beta
        self.gama = gama
        self.numbers_beta_change = numbers_bata_change
        self.numbers_every_miu = numbers_every_miu

        self.numbers_of_node = nx.number_of_nodes(G=Graph)

        self.lambda_c_QMF, self.lambda_c_HMF = threshold(G=self.Graph)


        adj = nx.to_numpy_array(G=Graph)
        self.adj = adj

        self.file_path = file_path
        self.group_name = group_name

    def multiprocess_once_to_steady(
        self,
        beta:float,
    ):

        label = 0
        if beta > self.lambda_c_HMF:
            label = 1
        elif beta == self.lambda_c_HMF:
            label = np.random.choice(a=1, size=1, replace=False).item()


        rng = np.random.default_rng()
        if self.ratio is None:
            infected_set = rng.choice(a=self.numbers_of_node, size=1, replace=False)
        else:
            infected_set = rng.choice(a=self.numbers_of_node, size=int(self.numbers_of_node * self.ratio), replace=False)

        state = np.zeros(shape=self.numbers_of_node)
        infected_set_to_save = infected_set
        # 每次感染之前保留当前已经感染的状态
        for _ in range(self.steps_to_steady):
            neighbors = []
            for node in infected_set:
                neighbors.extend(list(self.Graph.neighbors(node)))
            at_risk = np.array(neighbors)
            at_risk = np.unique(ar=at_risk)

            # recover
            all_idx = []
            for idx in range(len(infected_set)):
                if np.random.random() < self.gama:
                    all_idx.append(idx)
            
            infected_set = np.delete(arr=infected_set, obj=all_idx)

            # propagate
            for at_risk_node in at_risk:
                if np.random.random() < beta:
                    infected_set = np.append(infected_set, at_risk_node)
                    
            # 准稳态方法，始终保持有至少一个节点处于感染状态
            if len(infected_set) == 0:
                infected_set = infected_set_to_save
            else:
                infected_set_to_save = infected_set
        
        
        state[infected_set] = 1

        cur_data = np.insert(arr=state, obj=0, values=beta / self.gama)
        cur_data = np.append(arr=cur_data, values=label)

        return cur_data
    

    def multiprocess_simulation(self):
        """
        There is no return. But the dynamics data have saved in the file path what have gived in class.__init__()
        """

        Beta = np.linspace(
            start=self.beta[0],
            stop=self.beta[-1],
            num=self.numbers_beta_change,
        )
        with h5py.File(self.file_path + '.hdf5', 'a') as f:
            try:
                del f[self.group_name]
                cur_group = f.create_group(self.group_name)
            except KeyError:
                cur_group = f.create_group(self.group_name)

            # 保存邻接矩阵
            adjustment_dataset = cur_group.create_dataset(name='adj', data=self.adj, compression='gzip')
            data_dataset = cur_group.create_dataset(name='data', shape=(0, self.numbers_of_node + 2), maxshape=(None, self.numbers_of_node + 2), compression='gzip', dtype='f')
            pool = Pool(8)
            for _ in range(self.numbers_every_miu):
                results = pool.map(self.multiprocess_once_to_steady, Beta)
                results = np.array(results)
                data_dataset.resize((data_dataset.shape[0] + results.shape[0], self.numbers_of_node + 2))
                data_dataset[-results.shape[0]:, :] = results
            pool.close()
            pool.join()

            f.close()

