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
    begin_time = time.time()
    args = parameter_parser()
    group_name = f'Erdos_Renyi_Graph_N={args.N}'
    model_path=os.path.join(args.file_path, group_name, args.model_path)

    real_natworks_path = 'Real_Networks'
    real_results = 'Real_Results'
    times = 30

    edge_csv = np.loadtxt(os.path.join(real_natworks_path, 'tij_SFHH.dat'), dtype=int)
    edge_list = edge_csv[:, 1:]
    G = nx.from_edgelist(edgelist=edge_list)
    # G = nx.read_gml(os.path.join(real_natworks_path, 'power.gml'), label=None)
    # a = nx.number_of_nodes(G=G)
    # 删除孤立节点，保证输出结果的稳定性
    print(nx.number_of_nodes(G=G), nx.number_of_edges(G=G))
    G.remove_nodes_from(list(nx.isolates(G)))
    G = nx.convert_node_labels_to_integers(G=G, ordering='default')

    threshold_QMF, threshold_HMF = threshold(G=G)
    cur_prime = []
    # for i in tqdm(range(times)):
    cur_group_name = f'SFHH'
    trainer = Trainer(args=args, group_name=cur_group_name)
    SQS = DynamicsSIS(
        Graph=G,
        beta=(0.0001, 0.03),
        file_path=args.file_path,
        group_name=cur_group_name,
        numbers_every_miu=5000,
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
    cur_prime.append(lambda_c_pre)
    # lambda_c_pre = np.mean(cur_prime)
    print(f'The threshold get by nn, QMF, HMF are:{lambda_c_pre}、 {threshold_QMF}、{threshold_HMF} \n')

    with h5py.File(args.file_path + '.hdf5', 'r') as f:
        cur_group = f[cur_group_name]
        all_data = cur_group['data']
        df = pd.DataFrame(index=all_data[:, 0], data=all_data[:, 1:-1])

        all_idx = set(df.index.values)
        mean = {}
        for idx in all_idx:
            cur_data = df.loc[idx].values
            data_mean = np.mean(np.sum(cur_data, axis=-1))
            mean[idx] = data_mean
            df_mean = pd.Series(data=mean).sort_index()

        f.close()
        
    with plt.style.context(['science', 'ieee']):
        figure = plt.figure()
        plt.title(r'The results on SFHH network')
        ax = plt.subplot()
        ax.plot(df_out.index.values, df_out.values, color='blue', label='output by NN', ls='-.')
        ax.plot(df_mean.index.values, df_mean.values, color='red', label='Maximum Likelihood', ls='-')
        ax.axvline(x=threshold_QMF, ls='--', color='black',
                label=r'$\lambda_c^{QMF}$=' + str(round(threshold_QMF, 5)))
        ax.axvline(x=threshold_HMF, ls='--', color='black',
                label=r'$\lambda_c^{HMF}$=' + str(round(threshold_HMF, 5)))
        ax.axvline(x=lambda_c_pre, ls='--', color='black',
                label=r'$\lambda_c^\prime$=' + str(round(lambda_c_pre, 5)))
        # plt.xscale('log')

        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$\rho$')

        plt.legend()
        try:
            plt.savefig(os.path.join(real_results, cur_group_name + '_mean.png'))
        except OSError as e:
            os.mkdir(real_results)
            plt.savefig(os.path.join(real_results, cur_group_name + '_mean.png'))
        
        

        figure = plt.figure()
        plt.title(r'Output of I for every $\lambda$')
        plt.plot(df_var.index.values, df_var.values, color='blue', ls='--', label=r'I($\lambda$)')
        plt.axvline(x=threshold_QMF, ls='--', color='black',
                label=r'$\lambda_c^{QMF}$=' + str(round(threshold_QMF, 3)))
        plt.axvline(x=threshold_HMF, ls='--', color='black',
                label=r'$\lambda_c^{HMF}$=' + str(round(threshold_HMF, 3)))
        plt.axvline(x=lambda_c_pre, ls='--', color='black',
                label=r'$\lambda_c^\prime$=' + str(round(lambda_c_pre, 3)))
        
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'I($\lambda$)')

        plt.legend()
        plt.savefig(os.path.join(real_results, cur_group_name + '_var.png'))

        plt.close()

    end_time = time.time()

    totle_time = round(end_time - begin_time, 3)
    print('The time total cost on this work is:{}'.format(totle_time))

if __name__ == '__main__':
    main()