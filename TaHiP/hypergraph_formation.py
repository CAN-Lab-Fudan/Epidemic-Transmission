import numpy as np
import torch
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
# from dataloader import real_graph_loader


# T_max = 1.5
# n_real = 1501
# delta_t = T_max/(n_real-1)
#
# N = 100
# E_list = [500, 1000, 2000]
# b_list = [0.5, 0.75, 1, 1.25, 1.5]
# p_list = [0.1, 0.25, 0.5, 0.75, 0.9]
#
# key = [0, 0, 2]
# name = str(key[0])+str(key[1])+str(key[2])
#
# E = E_list[key[0]]
# # beta = b_list[key[1]]
# beta = 0.5           # threshold model
# thresh = 0.18
# title = "E"+str(E)+"_beta"+str(beta)
#
#
# # t_real2 = np.linspace(0, T_max*100, n_real)
#
#
# # def hghodr1(x, t, L):                            # product model
# #     x = torch.tensor(x, dtype=torch.float)
# #     a = -curing*x + beta * torch.matmul(L, torch.prod((x ** L.T).T, axis=0)) * (1 / x - 1)
# #     return a


def get_uniform_L(N, E, edge_size):
    hyperdegree = np.ones(N)
    hyperedges = []
    L = torch.zeros((N, E))
    N_edge = 0

    while N_edge < E:
        temp = np.random.choice(range(N), size=edge_size, replace=False,
                                p=hyperdegree / np.sum(hyperdegree))  # pa
        # temp = np.random.choice(range(N), size=edge_size, replace=False)      # random
        temp = set(temp)
        # if not any([(len(temp) == len(_e)) and np.all(temp == _e) for _e in hyperedges]):
        if temp not in hyperedges:
            hyperedges.append(temp)
            # edge_list.append(len(temp))
            hyperdegree[list(temp)] += 1
            L[list(temp), N_edge] = 1
            N_edge += 1
    return L


def get_L(N, E, lamda, real_graph, max_size):  # lamda of poisson distribution
    N_edge = 0
    hyperedges = []
    hyperdegree = np.zeros(N)
    L = torch.zeros((N, E))
    edge_list = []
    m = torch.distributions.Poisson(torch.tensor([lamda + 0.0]))
    if real_graph == 1:                  # just generate edge_list for realworld graph
        while N_edge < E:
            s = int(m.sample().item())
            if 2 <= s <= max_size:
                edge_list.append(s)
                N_edge += 1
        # dist = []
        # dist.append(edge_list.count(2) / len(edge_list))
        # dist.append(edge_list.count(3) / len(edge_list))
        # dist.append(edge_list.count(4) / len(edge_list))
        # dist.append(edge_list.count(5) / len(edge_list))
        return edge_list
    elif real_graph == 0:                          # if generate the random hypergraphs
        # specifically defined hyperedge size distribution %d=2, %d=3, %d=4, %d=5
        hs_d = [0.6, 0.3, 0.099, 0.001]                 # mean=2.5
        hs_d = [0.703, 0.267, 0.028, 0.0009]
        e_b = []        # edge number of different size
        e_b.append(int(E*hs_d[0]))
        e_b.append(int(E*hs_d[1]))
        e_b.append(int(E*hs_d[2]))
        e_b.append(E-e_b[0]-e_b[1]-e_b[2])
        e_s = []        # edge size
        for i in range(E):
            if i < e_b[0]:
                e_s.append(2)
            if e_b[0] <= i < e_b[1]+e_b[0]:
                e_s.append(3)
            if e_b[1]+e_b[0] <= i < e_b[2]+e_b[1]+e_b[0]:
                e_s.append(4)
            if e_b[2]+e_b[1]+e_b[0] <= i < sum(e_b):
                e_s.append(5)
        while N_edge < E:
            temp = np.random.choice(range(N), size=e_s[N_edge], replace=False,
                                    p=hyperdegree / np.sum(hyperdegree))          # pa
            # temp = np.random.choice(range(N), size=e_s[N_edge], replace=False)      # random
            temp = set(temp)
            # if not any([(len(temp) == len(_e)) and np.all(temp == _e) for _e in hyperedges]):
            if temp not in hyperedges:
                hyperedges.append(temp)
                # edge_list.append(len(temp))
                hyperdegree[list(temp)] += 1
                L[list(temp), N_edge] = 1
                N_edge += 1
        # torch.save(L, './data/Synthesized_hypergraph/highschool_pa_sythed')
        torch.save(L, './data/Synthesized_hypergraph/highschool_random_sythed')
    elif real_graph == 2:
        # edge_list = torch.load('./data/Realworld_hypergraph/contact-high-school/hs-hyperdegree')
        # edges = torch.load('./data/Realworld_hypergraph/house-bills/house-bills-edges')
        edges = torch.load('./data/Realworld_hypergraph/senate-bills/senate-bills-edges')
        edge_list = [len(edge) for edge in edges]
        # t_h = copy.deepcopy(target_hyperdegree)        # for updating
        # target_edge_size = torch.load('./data/Realworld_hypergraph/contact-high-school/hs-hyperedge-size')
        # hd_sample = np.ones(N)
        # # target_hyperdegree = [3, 3, 2, 2, 1]
        # # target_edge_size = [3, 2, 2, 2, 2]
        # nodes = range(N)
        # for edge_size in target_edge_size:
        #     # picking nodes that still need more hyperdegree by pa
        #     probability = hd_sample/np.sum(hd_sample)
        #     temp = np.random.choice(a=nodes, size=edge_size, replace=False, p=probability)
        #     while set(temp) in hyperedges:
        #         temp = np.random.choice(a=nodes, size=edge_size, replace=False, p=probability)
        #     for node in list(temp):
        #         t_h[node] -= 1
        #         hyperdegree[node] += 1
        #         hd_sample[node] += 1
        #         if t_h[node] == 0:
        #             # nodes.remove(node)
        #             # instead of removing node from nodes(those reaches target hyperdegree)
        #             # just set the probability of being chosen to 0
        #             hd_sample[node] = 0
        # hyperdegree.sort()
        # target_hyperdegree.sort()
        # print('ok')
    else:
        pass

    return edge_list


def get_data_product(N, E, beta, T_max, n_real, curing, plot_data, real_graph, edge_size):
    data1 = torch.zeros([N, n_real])
    # data2 = torch.zeros([N, n_real])
    ini_c_rate = torch.zeros(N)
    obs_data = torch.ones(N)
    delta_t = T_max / (n_real - 1)
    if real_graph == 1:
        edge_list = torch.load('./data/Realworld_hypergraph/hypergraph/house-bills/house-bills-edges')
        E1 = len(edge_list)         # original hyperedge number

        L = torch.zeros([N, E1])
        new_hyperedge = []
        for i in range(E1):
            for node in edge_list[i]:
                L[node-1, i] = 1                # nodes in dataset start from 1
        #
        # while i < E:
        #     temp = np.random.choice(range(N), size=edge_size, replace=False)      # random
        #     temp = set(temp)
        #     if temp not in new_hyperedge:
        #         new_hyperedge.append(temp)
        #         L[list(temp), i] = 1
        #         i += 1
        #     else:
        #         print('ops')

        # edge distribution preserved, hyperdegree distribution reformed
        # L_ = torch.zeros([N, E])
        # for i in range(E):
        #     temp = np.random.choice(a=N, size=int(torch.sum(L[:, i])), replace=False)
        #     L_[list(temp), i] = 1
        # hyperdegree_ = torch.sum(L_, axis=1)
    # else:
        # L = get_L(N=N, E=E, lamda=lamda, real_graph=0, max_size=5)       # edge sizes set

        # L = get_uniform_L(N, E, edge_size=edge_size)
        # hyperdegree, indices = torch.sort(torch.sum(L, axis=1))
        # plt.figure()
        # plt.title('Hyperdegree distribution of '+str(edge_size)+'-uniform hypergraph')
        # plt.xlabel("Node")
        # plt.ylabel("Hyperdegree")
        # plt.tick_params(axis="y", direction="in", )
        # plt.tick_params(axis="x", direction="in", )
        # plt.scatter(np.arange(N), hyperdegree, c='k', s=5)
        # plt.savefig('./data/Synthesized_hypergraph/uniform/pa/'+str(edge_size)+'/hyperdegree_edgesize' + str(
        #     edge_size) + '.png')
        # plt.close()
        # #
        # # torch.save(hyperdegree, './data/Synthesized_hypergraph/uniform/random/'+str(edge_size)+'/hyperdegree_edgesize'+str(edge_size))
        # torch.save(hyperdegree, './data/Synthesized_hypergraph/uniform/pa/'+str(edge_size)+'/hyperdegree_edgesize'+str(edge_size))
        # # torch.save(L, './data/Synthesized_hypergraph/uniform/random/'+str(edge_size)+'/L_edgesize'+str(edge_size))
        # torch.save(L, './data/Synthesized_hypergraph/uniform/pa/'+str(edge_size)+'/L_edgesize'+str(edge_size))

        # L = torch.load('./data/Synthesized_hypergraph/uniform/pa/'+str(edge_size)+'/L_edgesize'+str(edge_size))
        # L = torch.load('./data/Synthesized_hypergraph/uniform/random/'+str(edge_size)+'/L_edgesize'+str(edge_size))
        # L = torch.load('./data/Synthesized_hypergraph/hybrid/mean_hyperedgesize_4/L_hsbased_edgesize' + str(edge_size))

    obs_data = torch.rand(N)*0.3
    data1[:, 0] = obs_data
    # data2[:, 0] = obs_data
    obs1 = obs_data
    # obs2 = obs_data
    for i in tqdm(range(1, n_real)):
        obs1 = (1 - curing * delta_t) * obs1 + delta_t * beta * torch.matmul(L, torch.prod((obs1 ** L.T).T, dim=0)) * (
                    1 / obs1 - 1)              # social contagion(product) model
        obs1 = torch.clamp(obs1, 0.0001, 0.9999)
        data1[:, i] = obs1

        # obs2 = (1 - curing * delta_t) * obs2 + delta_t * beta x* \
        #        torch.matmul(L.unsqueeze(1)*1.0,
        #                     (torch.prod((obs2**L.T).T, dim=0).unsqueeze(0).repeat(N, 1)
        #                      *(1-obs2).unsqueeze(1).repeat(1, 7818)/(obs2**L.T).T).unsqueeze(2)).squeeze(1).squeeze(1)
        # data2[:, i] = obs2

    # torch.save
    if plot_data:
        time = np.arange(n_real)
        # plt.subplot(1, 2, 1)
        # plt.title(title)
        plt.xlabel("Time t/T$_{max}$")
        plt.ylabel("Node state x$_i$(t)")
        plt.tick_params(axis="y", direction="in",)
        plt.tick_params(axis="x", direction="in",)
        ave = torch.mean(data1, axis=0)
        for i in range(N):
            if i % 2 == 0:
                plt.scatter(time, data1[i, :], c='darkgrey', s=0.05)
            # plt.scatter(time, ave, c='r', s=0.5)
        plt.scatter(time, ave, c='r', s=0.5)
        # fig, ax = plt.subplots()
        # ax.legend(handles=[data1, ave], labels=['节点状态', '节点均值'])

        # plt.subplot(1, 2, 2)
        # for i in range(N):
        #     plt.scatter(time, sol_real2[i, :], c='k', s=0.5)

        plt.show()
        plt.close()

    return data1



def get_data_threshold(N, E, beta, T_max, n_real, curing, lamda, threshold, plot_data=True):
    data = torch.zeros([N, n_real])
    ini_c_rate = torch.zeros(N)
    obs_data = torch.zeros(N)
    delta_t = T_max / (n_real - 1)
    L, edge_list = get_L(N, E, lamda)
    for i in range(N):
        # ini_c_rate[i] = curing                # not nodewise
        # obs_data[i] = i * 0.2 / (N - 1) + 0.1  # initial state
        obs_data[i] = 0.1
    data[:, 0] = obs_data
    obs = obs_data
    for i in range(1, n_real):
        # obs = (1 - curing * delta_t) * obs + delta_t * beta * torch.matmul(L, torch.prod((obs ** L.T).T, dim=0)) * (
        #             1 / obs - 1)
        # data[:, i] = obs

        temp = torch.matmul(obs, L)
        temp[temp < threshold] = 0
        temp[temp > threshold] = 1          # edge activation list, EX1
        obs = (1 - curing * delta_t) * obs + delta_t * beta * torch.matmul(L, temp)
        data[:, i] = obs

    if plot_data:
        time = np.arange(n_real)
        # plt.subplot(1, 2, 1)
        # plt.title(title)

        plt.xlabel("Time t/T$_{max}$")
        plt.ylabel("Node state x$_i$(t)")
        plt.tick_params(axis="y", direction="in",)
        plt.tick_params(axis="x", direction="in",)

        # ave = np.mean(data, axis=1)
        for i in range(N):
            plt.scatter(time, data[i, :], c='k', s=0.5)
            plt.scatter(time, ave, c='r', s=0.5)
        # plt.subplot(1, 2, 2)
        # for i in range(N):
        #     plt.scatter(time, sol_real2[i, :], c='k', s=0.5)

        plt.show()

    return data, L, edge_list

# get_L(N=327, E=7818, lamda=2, real_graph=2, max_size=5)
# L, edge_list = get_L(N=1000, E=10000, lamda=2, real_graph=0)
# get_dat  
