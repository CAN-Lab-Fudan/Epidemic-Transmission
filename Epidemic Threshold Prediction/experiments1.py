from utils import UCM_SF_network, threshold
from SQS_simulation import SQS_simulation
from dataset import Just_Test
from trainer import Trainer
from parameters import parameter_parser

import networkx as nx
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


def main():
    args = parameter_parser()
    gama_min = 2
    gama_max = 13
    all_gama = np.arange(start=gama_min, stop=gama_max, step=10)
    threshold_s = {}
    for gama in tqdm(all_gama):
        G = UCM_SF_network(N=args.N, gama=gama)
        threshold_QMF, threshold_HMF = threshold(G=G)
        group_name = f'UCM_SF_network_gama={gama}'
        trainer = Trainer(args=args, group_name=group_name)

        trainer.run_data(G=G, lambda_range=(threshold_HMF - 0.05, threshold_HMF + 0.05))

        lambda_c_pre = trainer.fit_model()


        print(f'The threshold get by nn, QMF, HMF are:{lambda_c_pre}、 {threshold_QMF}、{threshold_HMF} \n')
        threshold_s[gama] = [threshold_QMF, threshold_HMF, lambda_c_pre]

    df = pd.DataFrame(data=threshold_s).T.sort_index()
    if not os.path.exists(args.results_path):
            os.mkdir(args.results_path)
    df.to_csv(path_or_buf=os.path.join(args.results_path, 'experiment_diff_game.csv'))

    with plt.style.context(['science', 'ieee']):
        figure = plt.figure()
        plt.title(r'The results on UCM\_SF network with N={} for different $\gamma$'.format(args.N))
        ax = plt.subplot()
        ax.plot(df.index.values, df.iloc[:, 0], color='red', label=r'$\lambda_c^{QMF}$')
        ax.plot(df.index.values, df.iloc[:, 1], color='blue', label=r'$\lambda_c^{HMF}$')
        ax.plot(df.index.values, df.iloc[:, 2], color='green', label=r'$\lambda_c^\prime$')

        # plt.yscale('log')

        plt.xlabel(r'$\gamma$')
        plt.ylabel(r'$\lambda_c$')

        plt.legend()
        plt.savefig(os.path.join(args.results_path, f'diff_gama.png'))

        plt.close()


if __name__ == '__main__':
    main()
