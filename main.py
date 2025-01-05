import networkx as nx
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from parameters import parameter_parser
from trainer import Trainer
from utils import threshold, threshold_SQS


def main():
    args = parameter_parser()
    G = nx.barabasi_albert_graph(n=args.N, m=args.k)
    trainer = Trainer(args=args, group_name=args.group_name)

    begin_time = time.time()
    # trainer.run_data(G=G, lambda_range=args.lambda_range)
    rundata_time = time.time()
    print(f'\n Run data cost time: {rundata_time - begin_time} \n')


    lambda_c_pre, df = trainer.fit_model()
    if not os.path.exists(args.results_path):
        os.mkdir(args.results_path)
    df.to_csv(os.path.join(args.results_path, 'output.csv'))
    train_time = time.time()
    print(f'Time cost on train model is: {train_time-rundata_time} \n')

    threshold_QMF, threshold_HMF = threshold(G=G)
    # threshold_sqs = threshold_SQS(args=args, group_name=args.group_name)

    print(f'The threshold get by nn, QMF, HMF, SQS are:{lambda_c_pre}、 {threshold_QMF}、{threshold_HMF} \n')


    with plt.style.context(['science', 'ieee']):
        figure = plt.figure()
        plt.title(f'The results on BA network with N={args.N}, k={args.k}')
        ax = plt.subplot()
        ax.plot(df.index.values, df.values, color='blue')

        ax.axvline(x=threshold_QMF, ls='--', color='black',
                   label=r'$\lambda_c^{QMF}$=' + str(round(threshold_QMF, 3)))
        ax.axvline(x=threshold_HMF, ls='--', color='black',
                   label=r'$\lambda_c^{HMF}$=' + str(round(threshold_HMF, 3)))
        ax.axvline(x=lambda_c_pre, ls='--', color='black',
                   label=r'$\lambda_c^\prime$=' + str(round(lambda_c_pre, 3)))
        # ax.axvline(x=threshold_sqs, ls='--', color='black',
        #           label=r'$\lambda_c^\prime$=' + str(round(threshold_sqs, 3)))
        plt.xscale('log')

        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$\rho$')

        plt.legend()
        plt.savefig(os.path.join(args.results_path, 'train_data_threshold.png'))

        plt.close()



if __name__ == '__main__':
    main()
