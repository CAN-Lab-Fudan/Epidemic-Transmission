import networkx as nx
import time
import pandas as pd
import numpy as np
import os
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import scienceplots

from parameters import parameter_parser
from trainer import Trainer
from utils import threshold, threshold_SQS, ER_Random_Graph
from dataset import Just_Test
from SQS_simulation import DynamicsSIS



# 改变ER图的平均度
def main():
    args = parameter_parser()
    N = args.N
    avg_k = 6
    group_name = f'Erdos_Renyi_Graph_N={N}_avg_k={avg_k}'
    model_path=os.path.join(args.file_path, group_name, args.model_path)
    N = args.N
    avg_k_min = 2
    avg_k_max = 12
    times = 30
    all_avg_k = np.arange(start=avg_k_min, stop=avg_k_max, step=1)
    threshold_s = {}
    for cur_avg_k in tqdm(all_avg_k):
        cur_c = []
        lambda_HMF = []
        lambda_QMF = []
        for i in tqdm(range(times)):
            G = ER_Random_Graph(N=N, avg_k=cur_avg_k)
            threshold_QMF, threshold_HMF = threshold(G=G)
            lambda_HMF.append(threshold_HMF)
            lambda_QMF.append(threshold_QMF)
            cur_group_name = f'Erdos_Renyi_Graph_average_k={cur_avg_k}'
            trainer = Trainer(args=args, group_name=cur_group_name)
            SQS = DynamicsSIS(
                Graph=G,
                beta=(max(0.01, threshold_HMF-0.2), min(threshold_HMF+0.2, 1)),
                file_path=args.file_path,
                group_name=cur_group_name,
                numbers_every_miu=30,
            )
            SQS.multiprocess_simulation()
            just = Just_Test(
                file_path=args.file_path,
                dataset_path=cur_group_name,
            )
            lambda_c_pre, df_out, df_var = trainer.test_and_get_threshold(
                model_path=model_path,
                group_name=cur_group_name,
            )
            cur_c.append(lambda_c_pre)
        lambda_c_pre = np.mean(cur_c)
        threshold_HMF = np.mean(lambda_HMF)
        threshold_QMF = np.mean(lambda_QMF)

        print(f'The threshold get by nn, QMF, HMF are:{lambda_c_pre}、 {threshold_QMF}、{threshold_HMF} \n')
        threshold_s[cur_avg_k] = [threshold_QMF, threshold_HMF, lambda_c_pre]


    df = pd.DataFrame(data=threshold_s).T.sort_index()
    if not os.path.exists(args.results_path):
            os.mkdir(args.results_path)
    df.to_csv(path_or_buf=os.path.join(args.results_path, 'ER_diff_average_k_by_6.csv'))


    # 画图
    all_index = df.index.values
    all_QMF = df.iloc[:, 0].values
    all_HMF = df.iloc[:, 1].values
    all_prime = df.iloc[:, -1].values
    RE_HMF = np.abs(all_HMF - all_prime) / all_HMF
    RE_QMF = np.abs(all_QMF - all_prime) / all_QMF

    with plt.style.context(['science', 'ieee']):
        figure = plt.figure()
        # plt.title(r'The results on ER graph with $N={}$ and $\bar k \in (2,14)$'.format(args.N))
        ax = plt.subplot()
        ax.plot(df.index.values, df.iloc[:, 0], color='red', marker='o', label=r'$\lambda_c^{QMF}$')
        ax.plot(df.index.values, df.iloc[:, 1], color='blue', marker='v', label=r'$\lambda_c^{HMF}$')
        ax.plot(df.index.values, df.iloc[:, -1], color='green', marker='>', label=r'$\lambda_c^\prime$')

        # plt.yscale('log')

        plt.xlabel(r'$\bar k$')
        plt.ylabel(r'$\lambda_c$')

        plt.legend()
        plt.savefig(os.path.join(args.results_path, 'ER_diff_average_k_by_6.png'))

        plt.figure()
        ax = plt.subplot()
        plt.title(r'Relative error between $\lambda_c^\prime$ and $\lambda_c^{HMF}$ or $\lambda_c^{QMF}$')
        ax.plot(all_index, RE_HMF, label=r'RE($\lambda_c^\prime$, $\lambda_c^{HMF}$)', color='blue', marker='v')
        ax.plot(all_index, RE_QMF, label=r'RE$\lambda_c^\prime$,$\lambda_c^{QMF}$)', color='red', marker='o')
        # plt.yscale('log')

        plt.xlabel(r'$\bar k$')
        plt.ylabel('Relative Error')

        plt.legend()
        plt.savefig(os.path.join(args.results_path, f'RE_ER_diff_average_k_by_6.png'))

        plt.close()
    pass



if __name__ == '__main__':
    main()