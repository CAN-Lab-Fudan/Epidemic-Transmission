import h5py
import networkx as nx
import pandas as pd
import numpy as np

import os
import os.path as osp
from utils import threshold
from parameters import parameter_parser

import matplotlib.pyplot as plt
import scienceplots

# 编写函数将数字转换为科学计数法的 LaTeX 格式
def format_sci_notation(num):
    exponent = int(np.floor(np.log10(abs(num))))
    mantissa = num / 10**exponent
    return r'${:.1f} \times 10^{{{}}}$'.format(mantissa, exponent)

if __name__ == "__main__":
    args = parameter_parser()
    N, avg_k = args.N, 4
    # G = ER_Random_Graph(N=N, avg_k=avg_k)
    group_name = f'Erdos_Renyi_Graph_N={N}'
    # group_name = f'Erdos_Renyi_Graph_N={N}_avg_k={avg_k}'
    # 读取所有的数据，将其放在一个名为all_data的array中，其中第0行表示对应的有效传播率lambda。
    with h5py.File(args.file_path + '.hdf5', 'r') as f:
        cur_group = f[group_name]
        adj = np.array(cur_group['adj'])

        all_data = cur_group['data']
        all_data = np.array(all_data[:, :])
        f.close()

    G = nx.from_numpy_array(A=adj)
    threshold_QMF, threshold_HMF = threshold(G=G)
    df = pd.DataFrame(index=all_data[:, 0], data=all_data[:, 1:-1])

    all_idx = set(df.index.values)
    mean = {}
    for idx in all_idx:
        cur_data = df.loc[idx].values
        data_mean = np.mean(np.sum(cur_data, axis=-1))
        mean[idx] = data_mean
    df_mean = pd.Series(data=mean).sort_index()

    # 对数据集进行k折交叉验证并使用保存最后的模型，模型保存的路径为 ./cross_validation/model_{idx}.pt
    model_folder = "./cross_validation"
    os.makedirs(model_folder, exist_ok=True)

    idx = 10
    # lambda_pre = [0.096363634, 0.105959594, 0.10020202, 0.10787879, 0.11747475, 0.12323232, 0.10979798, 0.11939394, 0.10404041, 0.115555555]
    # np.savetxt(fname=osp.join(model_folder, "thresholds.txt"), X=np.array(lambda_pre))
    lambda_pre = np.loadtxt(osp.join(model_folder, "thresholds.txt"))
    # 作图，展示结果的变化
    mse = np.mean((np.array(lambda_pre) - threshold_HMF)**2)
    with plt.style.context(['science', 'ieee']):
        figure = plt.figure()
        # plt.title(r'The Fluctuation on ER Graph with (N={},$\bar k$={})'.format(args.N, avg_k))
        plt.plot(np.arange(idx)+1, lambda_pre, ls=":", marker='o', color='blue', label=r'$\lambda_c^\prime$')
        # plt.plot(x, avg_sqs, ls=':', color='green', label=r'$\lambda_c^{SQS}$')
        plt.axhline(y=threshold_HMF, ls='--', color='black', label=r'$\lambda_c^{HMF}$')
        plt.axhline(y=threshold_QMF, ls='-.', color='red', label=r'$\lambda_c^{QMF}$')
        # plt.yscale("log")
        plt.xlabel(r'$k$')
        plt.ylabel(r'$\lambda$')
        plt.ylim((0, 0.2))
        s = format_sci_notation(mse)
        plt.text(0.8, 0.2, "MSE"+"={}".format(format_sci_notation(mse)), horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.legend()
        plt.savefig(osp.join(model_folder, "threshold.jpg"))
        plt.show()
        plt.close()