import networkx as nx
import numpy as np
from typing import Tuple, Union
import os
import pandas as pd
import h5py


def threshold(G: nx.Graph) -> Tuple[float, float]:
    adj = nx.to_numpy_array(G=G)
    # 理论计算的值
    # QMF
    lambda_all = np.linalg.eigvals(a=adj)
    lambda_c_QMF = 1 / np.max(lambda_all.real)

    # HMF
    degree = nx.degree_histogram(G=G)
    d_length = len(degree)
    d_sum = np.sum(np.array(degree))
    d_f, d_s = 0, 0
    for idx in range(d_length):
        d_f += idx * degree[idx] / d_sum
        d_s += pow(idx, 2) * degree[idx] / d_sum
    lambda_c_HMF = d_f / d_s

    return lambda_c_QMF, lambda_c_HMF


# 定义通用无标度网络
def UCM_SF_network(
        N: int,
        gama: Union[float, int],
        m: int = 2,
) -> nx.Graph:
    # 初始化网络结构，一个全联通的规则网络防止孤立节点
    Graph1 = nx.random_regular_graph(d=m, n=N)

    k_min= m
    k_max = np.sqrt(N)

    # 生成每个节点的度，抽样方法
    degrees = np.empty(shape=N, dtype=int)
    idx = 0
    while True:
        seed = np.random.zipf(a=gama)
        if k_min <= seed <= k_max:
            degrees[idx] = seed
            idx += 1
        if idx == N:
            break

    # 若是总的度不为偶数，则随机挑选进行更换
    while np.sum(degrees) % 2 != 0:
        seed = np.random.zipf(a=gama)
        if m <= seed <= np.sqrt(N):
            idx = np.random.choice(a=np.arange(N), size=1)
            degrees[idx] = seed

    Graph2 = nx.expected_degree_graph(w=degrees)

    Graph = nx.compose(G=Graph2, H=Graph1)

    Graph = nx.convert_node_labels_to_integers(G=Graph, ordering='default')


    return Graph


def ER_Random_Graph(N: int, avg_k: int) -> nx.Graph:
    m = avg_k * N
    G = nx.gnm_random_graph(n=N, m=m)
    G.remove_nodes_from(list(nx.isolates(G)))
    G = nx.convert_node_labels_to_integers(G=G, ordering='default')

    return G


def threshold_SQS(file_path: str, group_name: str) -> float:
    with h5py.File(file_path + '.hdf5', 'r') as f:
        cur_group = f[group_name]

        all_data = cur_group['data']
        adj = np.array(cur_group['adj'])
        df = pd.DataFrame(index=all_data[:, 0], data=all_data[:, 1:-1])
        df_sum = df.sum(axis=1)
        # df_mean = df.mean(axis=1)
        N, _ = adj.shape

        f.close()
    all_sqs = {}
    for idx in set(df.index.values):
        values = df_sum.loc[idx]
        sqs = (np.mean(values ** 2)  - np.mean(values) ** 2) / np.mean(values)
        # sqs = np.std(values) / np.mean(values)
        all_sqs[idx] = sqs
    df_sqs = pd.Series(all_sqs).sort_index()

    lambda_c_sqs = float(df_sqs.idxmax())

    return lambda_c_sqs


def random_reconnet(Graph: nx.Graph, ratio: float) -> nx.Graph:
    N = nx.number_of_nodes(G=Graph)
    numbers_edges = nx.number_of_edges(G=Graph)
    reconnect_numbers = int(numbers_edges * ratio)
    edges_set = list(Graph.edges)
    remove_idx = np.random.choice(a=numbers_edges, size=reconnect_numbers, replace=False)
    remove_edge_list = []
    for i in range(reconnect_numbers):
        idx = remove_idx[i]
        remove_edge_list.append(edges_set[idx])
    Graph.remove_edges_from(remove_edge_list)

    # 随机增加边，数目与移除边的数目相等
    numbers_add_edge = reconnect_numbers
    while numbers_add_edge > 0:
        nodes = np.random.choice(a=N, size=2, replace=False)
        if not Graph.has_edge(nodes[0], nodes[-1]):
            Graph.add_edge(u_of_edge=nodes[0], v_of_edge=nodes[-1])
            numbers_add_edge -= 1
    
    # 删除孤立节点，保证输出结果的稳定性
    Graph.remove_nodes_from(list(nx.isolates(Graph)))
    Graph = nx.convert_node_labels_to_integers(G=Graph, ordering='default')

    return Graph



