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


def main():
    args = parameter_parser()
    N = args.N
    avg_k_min = 4
    avg_k_max = 24
    avg_k = 4
    N_min = 500
    N_max = 5100
    df = pd.read_csv(filepath_or_buffer=os.path.join(args.results_path, 'ER_diff_N.csv'))
    # 画图
    all_index = df.iloc[:, 0].values
    all_QMF = df.iloc[:, 1].values
    all_HMF = df.iloc[:, 2].values
    all_prime = df.iloc[:, -1].values

    RE_HMF = np.abs(all_HMF - all_prime) / all_HMF
    RE_QMF = np.abs(all_QMF - all_prime) / all_QMF


    with plt.style.context(['science', 'ieee']):
        figure = plt.figure()
        plt.title(r'Results on ER Graph ($N \in ({},{})$, $\bar k={}$)'.format(N_min, N_max - 100, avg_k))
        ax = plt.subplot()
        ax.plot(all_index, all_QMF, color='red', marker='o', label=r'$\lambda_c^{QMF}$')
        ax.plot(all_index, all_HMF, color='blue', marker='v', label=r'$\lambda_c^{HMF}$')
        ax.plot(all_index, all_prime, color='green', marker='>', label=r'$\lambda_c^\prime$')

        # plt.yscale('log')

        plt.xlabel(r'N')
        plt.ylabel(r'$\lambda_c$')
        plt.ylim((0.05, 0.15))

        plt.legend()
        plt.savefig(os.path.join(args.results_path, 'ER_diff_N.png'))


        plt.figure()
        ax = plt.subplot()
        plt.title(r'Relative error between $\lambda_c^\prime$ and $\lambda_c^{HMF}$ or $\lambda_c^{QMF}$')
        ax.plot(all_index, RE_HMF, label=r'RE($\lambda_c^\prime$, $\lambda_c^{HMF}$)', color='blue', marker='v')
        ax.plot(all_index, RE_QMF, label=r'RE$\lambda_c^\prime$,$\lambda_c^{QMF}$)', color='red', marker='o')
        # plt.yscale('log')

        plt.xlabel(r'N')
        plt.ylabel('Relative Error')

        plt.legend()
        plt.savefig(os.path.join(args.results_path, f'RE_ER_diff_N.png'))

        plt.close()


if __name__ == '__main__':
    main()