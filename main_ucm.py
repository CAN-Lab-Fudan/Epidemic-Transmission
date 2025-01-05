import networkx as nx
import time
import os
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from parameters import parameter_parser
from trainer import Trainer
from utils import threshold, threshold_SQS, UCM_SF_network


def main():
    args = parameter_parser()
    gama = 2.25
    G = UCM_SF_network(N=args.N, gama=gama)
    threshold_QMF, threshold_HMF = threshold(G=G)
    group_name = f'UCM_SF_graph_N={args.N}'
    trainer = Trainer(args=args, group_name=group_name)

    begin_time = time.time()
    trainer.run_data(G=G, lambda_range=(0.01, 2 * threshold_HMF))
    rundata_time = time.time()
    print(f'\n Run data cost time: {rundata_time - begin_time} \n')


    lambda_c_pre, df_out, df_var = trainer.fit_model()
    if not os.path.exists(args.results_path):
        os.mkdir(args.results_path)
    df_out.to_csv(os.path.join(args.results_path, group_name + '_output.csv'))
    train_time = time.time()
    print(f'Time cost on train model is: {train_time-rundata_time} \n')

    with h5py.File(args.file_path + '.hdf5', 'r') as f:
        cur_group = f[group_name]
        adj = np.array(cur_group['adj'])
        G = nx.from_numpy_array(A=adj)
        threshold_QMF, threshold_HMF = threshold(G=G)
        all_data = cur_group['data']
        df = pd.DataFrame(index=all_data[:, 0], data=all_data[:, 1:-1])

        all_idx = set(df.index.values)
        mean = {}
        for idx in all_idx:
            cur_data = df.loc[idx].values
            data_mean = np.mean(np.sum(cur_data, axis=-1))
            # data_mean = np.mean(cur_data)
            mean[idx] = data_mean
        df_mean = pd.Series(data=mean).sort_index()

        f.close()

    print(f'The threshold get by nn, QMF, HMF, SQS are:{lambda_c_pre}、 {threshold_QMF}、{threshold_HMF}\n')
    threshold_sqs = threshold_SQS(file_path=args.file_path, group_name=group_name)


    with plt.style.context(['science', 'ieee']):
        figure = plt.figure()
        plt.title(f'The results on UCM\_SF network with N={args.N}, gama={gama}')
        ax = plt.subplot()
        ax.plot(df_out.index.values, df_out.values, color='blue', label='output by NN', ls='-.')
        ax.plot(df_mean.index.values, df_mean.values, color='red', label='Maximum Likelihood', ls='-')
        ax.axvline(x=threshold_QMF, ls='--', color='black',
                   label=r'$\lambda_c^{QMF}$=' + str(round(threshold_QMF, 3)))
        ax.axvline(x=threshold_HMF, ls='--', color='black',
                   label=r'$\lambda_c^{HMF}$=' + str(round(threshold_HMF, 3)))
        ax.axvline(x=lambda_c_pre, ls='--', color='black',
                   label=r'$\lambda_c^\prime$=' + str(round(lambda_c_pre, 3)))
        ax.axvline(x=threshold_sqs, ls='--', color='black',
                   label=r'$\lambda_c^{SQS}$=' + str(round(threshold_sqs, 3)))
        # plt.xscale('log')

        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$n$')

        plt.legend()
        plt.savefig(os.path.join(args.results_path, group_name + '_train.png'))

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
        plt.savefig(os.path.join(args.results_path, group_name + '_var.png'))

        plt.close()



if __name__ == '__main__':
    main()
