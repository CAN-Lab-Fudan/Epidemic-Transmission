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


# 比较各种不同的平均方式，首先第一种方式，生成一个固定的图，观察我们的预测值的波动情况
def main():
    args = parameter_parser()
    avg_k = 4
    gamma = 2.25
    group_name = f'Erdos_Renyi_Graph_N={args.N}'
    model_path=os.path.join(args.file_path, group_name, args.model_path)
    threshold_prime = []
    threshold_sqs = []
    avg_prime = []
    avg_sqs = []
    times = 101
    G = ER_Random_Graph(N=args.N, avg_k=avg_k)
    # G = UCM_SF_network(N=args.N, gama=gamma)
    threshold_QMF, threshold_HMF = threshold(G=G)
    for i in tqdm(range(times)):
        cur_group_name = f'ER_Graph_N={args.N}_avg_k={avg_k}'
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
        threshold_prime.append(lambda_c_pre)
        # lambda_c_sqs = 1
        # threshold_sqs.append(lambda_c_sqs)
        if i % 5 == 0:
            avg_prime.append(np.mean(threshold_prime))
            # avg_sqs.append(np.mean(threshold_sqs))

    
    x = np.arange(len(avg_prime)) * 5

    with plt.style.context(['science', 'ieee']):
        figure = plt.figure()
        # plt.title(r'The Fluctuation on ER Graph with (N={},$\bar k$={})'.format(args.N, avg_k))
        plt.plot(x, avg_prime, ls=":", marker='o', color='blue', label=r'$\lambda_c^\prime$')
        # plt.plot(x, avg_sqs, ls=':', color='green', label=r'$\lambda_c^{SQS}$')
        plt.axhline(y=threshold_HMF, ls='--', color='black', label=r'$\lambda_c^{HMF}$')
        plt.axhline(y=threshold_QMF, ls='-.', color='red', label=r'$\lambda_c^{QMF}$')

        plt.xlabel(r'$\eta$')
        plt.ylabel(r'$\lambda$')
        plt.ylim((0.05, 0.15))
        # plt.text(0.05, 0.95, '(b)', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.legend()
        try:
            plt.savefig(os.path.join('Validation_Average', 'same_network.png'))
        except OSError:
            os.mkdir('Validation_Average')
            plt.savefig(os.path.join('Validation_Average', 'same_network.png'))

        plt.close()


if __name__ == '__main__':
    main()
