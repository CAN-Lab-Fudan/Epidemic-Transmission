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
import scienceplots
import time
from tqdm import tqdm


def main():
    args = parameter_parser()
    N_min = 500
    N_max = 1500
    all_N = np.arange(start=N_min, stop=N_max, step=100)
    threshold_s = {}
    for n in tqdm(all_N):
        G = nx.erdos_renyi_graph(n=n, p=0.1)
        group_name = f'Erdos_Renyi_Graph_N={n}'
        trainer = Trainer(args=args, group_name=group_name)
        threshold_QMF, threshold_HMF = threshold(G=G)

        trainer.run_data(G=G, lambda_range=(max(0.01, threshold_HMF-0.2), min(threshold_HMF+0.2, 1)))

        lambda_c_pre, df_out, df_var = trainer.fit_model()

        print(f'The threshold get by nn, QMF, HMF are:{lambda_c_pre}、 {threshold_QMF}、{threshold_HMF} \n')
        threshold_s[n] = [threshold_QMF, threshold_HMF, lambda_c_pre]

        with plt.style.context(['science', 'ieee']):
            figure = plt.figure()
            plt.title(f'The results on ER network with N={n}')
            ax = plt.subplot()
            ax.plot(df_out.index.values, df_out.values, color='blue', label='output by NN', ls='-.')

            ax.axvline(x=threshold_QMF, ls='--', color='black',
                    label=r'$\lambda_c^{QMF}$=' + str(round(threshold_QMF, 3)))
            ax.axvline(x=threshold_HMF, ls='--', color='black',
                    label=r'$\lambda_c^{HMF}$=' + str(round(threshold_HMF, 3)))
            ax.axvline(x=lambda_c_pre, ls='--', color='black',
                    label=r'$\lambda_c^\prime$=' + str(round(lambda_c_pre, 3)))
            # plt.xscale('log')

            plt.xlabel(r'$\lambda$')
            plt.ylabel(r'$\rho$')

            plt.legend()
            if not os.path.exists('ER_N_with_train'):
                os.mkdir('ER_N')
            plt.savefig(os.path.join('ER_N', group_name + '_mean.png'))

            figure = plt.figure()
            plt.title(f'Susceptibility on the output of NN')
            plt.plot(df_var.index.values, df_var.values, color='blue', ls='--', label='Std/Mean')
            plt.axvline(x=threshold_QMF, ls='--', color='black',
                    label=r'$\lambda_c^{QMF}$=' + str(round(threshold_QMF, 3)))
            plt.axvline(x=threshold_HMF, ls='--', color='black',
                    label=r'$\lambda_c^{HMF}$=' + str(round(threshold_HMF, 3)))
            plt.axvline(x=lambda_c_pre, ls='--', color='black',
                    label=r'$\lambda_c^\prime$=' + str(round(lambda_c_pre, 3)))
        
            plt.xlabel(r'$\lambda$')
            plt.ylabel(r'std')

            plt.legend()
            plt.savefig(os.path.join('ER_N', group_name + '_var.png'))

            plt.close()



    df = pd.DataFrame(data=threshold_s).T.sort_index()
    if not os.path.exists(args.results_path):
        os.mkdir(args.results_path)
    df.to_csv(path_or_buf=os.path.join(args.results_path, 'experiment_diff_N_ER.csv'))

    with plt.style.context(['science', 'ieee']):
        figure = plt.figure()
        plt.title(r'The results on ER network')
        ax = plt.subplot()
        ax.plot(df.index.values, df.iloc[:, 0], color='red', label=r'$\lambda_c^{QMF}$')
        ax.plot(df.index.values, df.iloc[:, 1], color='blue', label=r'$\lambda_c^{HMF}$')
        ax.plot(df.index.values, df.iloc[:, 2], color='green', label=r'$\lambda_c^\prime$')

        # plt.yscale('log')

        plt.xlabel(r'N')
        plt.ylabel(r'$\lambda_c$')

        plt.legend()
        plt.savefig(os.path.join(args.results_path, 'diff_N_ER.png'))

        plt.close()


if __name__ == '__main__':
    main()
