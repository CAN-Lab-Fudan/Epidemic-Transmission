import networkx as nx
import time
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import scienceplots

from parameters import parameter_parser
from trainer import Trainer
from utils import threshold, threshold_SQS, UCM_SF_network, ER_Random_Graph
from dataset import Just_Test
from SQS_simulation import DynamicsSIS


# 验证数据不对成性对预测的影响
def main():
    args = parameter_parser()
    avg_k = 4
    gamma = 2.25
    group_name = f'Erdos_Renyi_Graph_N={args.N}'
    model_path=os.path.join(args.file_path, group_name, args.model_path)
    G = ER_Random_Graph(N=args.N, avg_k=avg_k)
    threshold_QMF, threshold_HMF = threshold(G=G)
    cur_group_name = f'ER_Graph_N={args.N}_avgk={avg_k}'
    times = 10
    offset_ratio = []
    avg_prime = []
    for i in tqdm(range(times)):
        beta=(max(0.01, threshold_HMF-0.1), min(threshold_HMF + 0.1 + i * 0.05, 1))
        offset_ratio.append((i * 0.05) / 0.1)
        repeat = 30
        threshold_prime = []
        for j in tqdm(range(repeat)):
            trainer = Trainer(args=args, group_name=cur_group_name)
            SQS = DynamicsSIS(
                Graph=G,
                beta=beta,
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
            threshold_prime.append(lambda_c_pre)

        avg_prime.append(np.mean(threshold_prime))

    with plt.style.context(['science', 'ieee']):
        figure = plt.figure()
        # plt.title(r'The Fluctuation on ER Graph (N={},$\bar k$={})'.format(args.N, avg_k))
        plt.plot(offset_ratio, avg_prime, ls=':', marker='o', color='blue', label=r'$\lambda_c^\prime$')
        plt.axhline(y=threshold_HMF, ls='--', color='black', label=r'$\lambda_c^{HMF}$')
        plt.axhline(y=threshold_QMF, ls='-.', color='red', label=r'$\lambda_c^{QMF}$')

        plt.xlabel(r'$\pi$')
        plt.ylabel(r'$\lambda$')

        plt.ylim((0.05, 0.15))
        plt.text(0.05, 0.95, '(a)', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

        plt.legend()
        try:
            plt.savefig(os.path.join('Validation_Average', 'diff_range.png'))
        except OSError:
            os.mkdir('Validation_Average')
            plt.savefig(os.path.join('Validation_Average', 'diff_range.png'))

        plt.close()


if __name__ == '__main__':
    main()
