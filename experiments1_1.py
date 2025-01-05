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
from utils import threshold, threshold_SQS, UCM_SF_network
from dataset import Just_Test
from SQS_simulation import DynamicsSIS


def main():
    args = parameter_parser()
    group_name = f'Erdos_Renyi_Graph_N={args.N}'
    # group_name = f'UCM_SF_graph_N={args.N}'
    model_path=os.path.join(args.file_path, group_name, args.model_path)
    gama_min = 2
    gama_max = 4
    times = 30
    all_gama = np.arange(start=gama_min, stop=gama_max, step=0.2)
    threshold_s = {}
    for gama in tqdm(all_gama):
        cur_c = []
        lambda_HMF = []
        lambda_QMF = []
        for i in range(times):
            G = UCM_SF_network(N=args.N, gama=gama)
            threshold_QMF, threshold_HMF = threshold(G=G)
            lambda_HMF.append(threshold_HMF)
            lambda_QMF.append(threshold_QMF)
            cur_group_name = f'UCM_SF_network_gama={gama}'
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
        threshold_s[gama] = [threshold_QMF, threshold_HMF, lambda_c_pre]


    df = pd.DataFrame(data=threshold_s).T.sort_index()
    try:
        df.to_csv(path_or_buf=os.path.join(args.results_path, 'UCM_SF_diff_game.csv'))
    except OSError:
        os.mkdir(args.results_path)
        df.to_csv(path_or_buf=os.path.join(args.results_path, 'UCM_SF_diff_game.csv'))

    # 画图
    all_index = df.index.values
    all_QMF = df.iloc[:, 0].values
    all_HMF = df.iloc[:, 1].values
    all_prime = df.iloc[:, -1].values
    RE_HMF = np.abs(all_HMF - all_prime) / all_HMF
    RE_QMF = np.abs(all_QMF - all_prime) / all_QMF
    with plt.style.context(['science', 'ieee']):
        figure = plt.figure()
        # plt.title(r'Results on UCM\_SF Graph (N={}, $\gamma \in ({},{})$)'.format(args.N, gama_min, gama_max))
        ax = plt.subplot()
        ax.plot(df.index.values, df.iloc[:, 0], color='red', marker='o', label=r'$\lambda_c^{QMF}$')
        ax.plot(df.index.values, df.iloc[:, 1], color='blue', marker='v', label=r'$\lambda_c^{HMF}$')
        ax.plot(df.index.values, df.iloc[:, -1], color='green', marker='>', label=r'$\lambda_c^\prime$')

        # plt.yscale('log')

        plt.xlabel(r'$\gamma$')
        plt.ylabel(r'$\lambda_c$')
        plt.text(0.95, 0.05, '(b)', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.legend()
        plt.savefig(os.path.join(args.results_path, f'UCM_SF_diff_gama.png'))

        plt.figure()
        ax = plt.subplot()
        plt.title(r'Relative Error Between $\lambda_c^\prime$ and $\lambda_c^{HMF}$ or $\lambda_c^{QMF}$')
        ax.plot(all_index, RE_HMF, label=r'RE($\lambda_c^\prime$, $\lambda_c^{HMF}$)', color='blue', marker='v')
        ax.plot(all_index, RE_QMF, label=r'RE$\lambda_c^\prime$,$\lambda_c^{QMF}$)', color='red', marker='o')
        # plt.yscale('log')

        plt.xlabel(r'$\gamma$')
        plt.ylabel('Relative Error')

        plt.legend()
        plt.savefig(os.path.join(args.results_path, f'RE_UCM_SF_diff_gamma.png'))

        plt.close()


if __name__ == '__main__':
    main()

        