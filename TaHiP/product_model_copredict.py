import sys
sys.path.append('..')
from typing import List, Iterator
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from trash.hypergraph_formation import get_data_product, get_L
import matplotlib.pyplot as plt



class HyperGraph(nn.Module):
    def __init__(self, N_node, N_edge, edge_dist: List[int], delta_t, curing, beta):
        # def __init__(self, N_node, N_edge, delta_t, delta, beta, seq_len):
        # edge_dist: 1xN_edge, a list containing the number of nodes in the edge, for example,
        # if `N_edge = 8`, then edge_list can be `[1, 1, 1, 2, 2, 3, 4, 4]`
        super(HyperGraph, self).__init__()

        # L = torch.load('./data/Synthesized_hypergraph/N500_E5000_evolving_L_2_5')
        # L_adj = L@L.T
        self.N_node = N_node
        self.N_edge = N_edge
        # self.register_buffer("edge_list", torch.tensor(edge_dist))
        self.delta_t = delta_t
        # self.ini = 0.01
        self.delta = curing

        # self.register_buffer('delta', delta)
        self.beta = beta

        # self.para_matrix = torch.nn.Parameter(torch.randn(N_node, N_edge))
        # self.para_matrix = torch.ones([N_node, N_edge])*self.ini
        # self.para_matrix = torch.zeros([N_node, N_edge])
        self.para_matrix = torch.zeros([N_node, N_edge])
        # zero_count = int(zero_ratio*N_node)
        # for i in range(N_edge):
        #     self.para_matrix[torch.randperm(N_node)[:zero_count], i] = 0
        # self.para_matrix = torch.sigmoid(torch.randn((N_node, N_edge)))

        # L = torch.load('./data/Synthesized_hypergraph/N500_E5000_evolving_L_2_5')
        # for i in range(2500):
        #     L[:,i] = torch.zeros(500)
        #     L[torch.randperm(N_node)[:edge_dist[i]], i] = 1
        # self.para_matrix = L
        # hyperdegree = np.ones(N_node)
        for i, edge in enumerate(edge_dist):
            # temp = np.random.choice(range(N_node), size=edge, replace=False,            # pa_ini
            #                         p=hyperdegree / np.sum(hyperdegree))
            # self.para_matrix[temp, i] = 1
            # hyperdegree[temp] += 1
            self.para_matrix[torch.randperm(N_node)[:edge], i] = 1
            # self.para_matrix[torch.randperm(N_node)[:edge], i] = 1+self.ini-self.ini*self.N_node/edge

        # self.para_matrix = torch.nn.Parameter(torch.sigmoid(self.para_matrix))

        # self.para_matrix = torch.load('./data/para_m')

        self.para_matrix = torch.nn.Parameter(self.para_matrix.float())

    def _sum_prod1(self, obs):
        L = self.para_matrix
        power_buffer = obs ** L                             # power_buffer (NXE)
        all_prod = torch.prod(power_buffer, dim=0)          # all_prod (E)
        # reduced_prod (NXE)
        reduced_prod = torch.stack([all_prod * (1 - obs[i]) / buf for i, buf in enumerate(power_buffer)], dim=0)
        # (NX1XE) @ (NXEX1) -> (NX1), to obtain relative sum of products of each node
        return torch.matmul(L.unsqueeze(1), reduced_prod.unsqueeze(2)).squeeze(1)

    def _sum_prod2(self, obs):
        L = self.para_matrix
        power_buffer = obs ** L                             # power_buffer (NXE)
        all_prod = torch.prod(power_buffer, dim=0)          # all_prod (E)
        obs_ = (1 - obs).repeat(1, self.N_edge)
        a = all_prod.unsqueeze(0).repeat(self.N_node, 1)
        r_p = a * obs_ / power_buffer
        # r_p = torch.prod(obs ** L, dim=0).unsqueeze(0).repeat(self.N_node, 1)*(1-obs).repeat(1, self.N_edge)/obs ** L
        # reduced_prod (NXE)
        # (NX1XE) @ (NXEX1) -> (NX1), to obtain relative sum of products of each node
        return torch.matmul(L.unsqueeze(1), r_p.unsqueeze(2)).squeeze(1)

    def forward(self, obs):
        # obs (NX1)
        # assert obs.dim() == 2 and obs.shape[1] == 1, f"Invalid input shape {obs.shape}."
        if obs.shape[1] != 1:
            obs.unsqueeze(1)
        L = self.para_matrix
        # next_obs = (1 - self.delta * self.delta_t) * obs + self.delta_t * self.beta * self.mlp(obs)
        # next_obs1 = (1 - self.delta * self.delta_t) * obs + self.delta_t * self.beta * self._sum_prod1(obs)
        next_obs2 = (1 - self.delta * self.delta_t) * obs + self.delta_t * self.beta * self._sum_prod2(obs)
        # next_obs = (1 - self.delta * self.delta_t) * obs + self.delta_t * self.beta * \
        #            torch.matmul(L, torch.prod(obs ** L, dim=0)).unsqueeze(1) * (1 / obs - 1)
        next_obs = torch.clamp(next_obs2, 0.01, 1 - 0.01)
        # next_obs = torch.sigmoid(next_obs)
        return next_obs

    # def hyper_count_loss(self):
    #     return ((torch.sum(self.para_matrix, dim=0) - self.edge_list) ** 2).sum() / self.para_matrix.numel()

    def predict_loss(self, pred, gt):
        # pred/gt: NXk (k is batch and k <= T)
        # if pred.shape != gt.shape:
        #     print("sa")
        # weighted loss, later loss greater than ealier loss
        # return (torch.sum((pred - gt) ** 2, dim=0)*self.train_loss_weights.cuda()).sum()/pred.shape[1]
        # return torch.mean((pred - gt) ** 2)
        return ((pred - gt) ** 2).sum() / pred.shape[1]


def train_in_batch(cur_train, graph, k, b, lc1, lc2, c):
    cnt = 0
    _t1 = cur_train[b, :, 0:1]
    # _t3 = cur_train[b, :, 0:1]
    # loss1 = torch.zeros(1, device=cur_train.device)
    loss2 = torch.zeros(1, device=cur_train.device)
    # print(b)
    while cnt < k:
        # _t1 = graph.forward(_t1)
        # _t3 = graph.forward(_t3)
        # if torch.sum(torch.isnan(_t1)) > 0:
        #     print("TRAIN_nan")
        #     break
        # loss1 += graph.predict_loss(_t1, cur_train[b, :, cnt + 1:cnt + 2])  # compute loss1 with t1' & gt

        t1 = graph.forward(cur_train[b, :, cnt:cnt + 1])  # bigger steps?
        loss2 += graph.predict_loss(t1, cur_train[b, :, cnt + 1:cnt + 2])  # compute loss2 with t1 & gt1

        # if cnt % int(k/4) == 0:
        #     _t3 = cur_train[b, :, cnt:cnt+1]
        # loss3 += graph.predict_loss(_t3, cur_train[b, :, cnt+1:cnt+2])
        cnt += 1
    # if b == 0 and c == 0:
    #     print(str(loss1 / loss2))

    # return (loss1 * lc1 + loss2 * lc2) / 10
    return loss2 * lc2    

@torch.no_grad()
def test(graph, data, N, T, epoch, exp_name, optimizer, test_idx, min):
    # test_data = test_data.T
    pred_loss = []
    # test_num = int(T*test_ratio)-1
    # train_num = T-test_num

    test_num = T  # full test
    preds = torch.zeros([N, test_num]).cuda()
    cur_pred = data[test_idx, :, 0:1].cuda()
    for t in range(test_num):
        cur_pred = graph.forward(cur_pred)
        # if torch.sum(torch.isnan(cur_pred)) > 0:
        #     print('TEST_nan')
        #     break
        preds[:, t] = cur_pred.squeeze(1)  # epoch wise
    preds = preds.cpu()
    real_avg = torch.sum(data[test_idx], axis=0)/N
    prd_avg = torch.sum(preds, axis=0)/N
    test_loss_mse = ((preds - data[test_idx, :, 1:]) ** 2).mean()
    test_loss_mae = torch.abs(preds - data[test_idx, :, 1:]).mean()
    # if test_loss_mse < min or epoch%60==0:
    #     time = np.arange(T)
    #     # time_test = np.arange(test_num)+train_num
    #     plt.figure(figsize=(12, 6))
    #     plt.subplot(1, 2, 1)
    #     # plt.title("Lr = " + str(optimizer.param_groups[0]['lr']) + 'MAE=' + str(test_loss_mae))
    #     plt.title('MAE=' + str(test_loss_mae))
    #     plt.xlabel("Time t/T$_{max}$")
    #     plt.ylabel("Node state x$_i$(t)")
    #     plt.tick_params(axis="y", direction="in", )
    #     plt.tick_params(axis="x", direction="in", )
    #     plt.scatter(time, real_avg[1:], c='g', s=0.3)
    #     plt.scatter(time, prd_avg, c='r', s=0.3)
    #     # ave = np.mean(data, axis=1)
    #     # for i in range(N):
    #     #     if i % 30 == 7:
    #     #         plt.scatter(time, data[test_idx, i, 1:], c='k', s=0.1)
    #     #         # plt.scatter(time_test, preds[i], c='r', s=0.1)
    #     #         plt.scatter(time, preds[i], c='r', s=0.1)
    #     plt.subplot(1, 2, 2)
    #     # plt.title('Selected nodes test_idx=' + str(test_idx))
    #     plt.title('MSE=' + str(test_loss_mse.detach().item()))
    #     plt.xlabel("Time t/T$_{max}$")
    #     plt.ylabel("Node state x$_i$(t)")
    #     plt.tick_params(axis="y", direction="in", )
    #     plt.tick_params(axis="x", direction="in", )
    #     list = [0, 1, 3, 6, 8, 18, 33, 155, 159, 189, 239, 324]  # high-school
    #     # list = [0, 3, 11, 20, 48, 76, 90, 110, 130]                                 # primary-school
    #     # list = [0, 3, 11, 20, 48, 76, 5, 90, 120, 150, 180]
    #     for i in list:
    #         plt.scatter(time, data[test_idx, i, 1:], c='k', s=0.5)
    #         plt.scatter(time, preds[i], c='r', s=0.5)
    #     # plt.show()
    #     if test_loss_mse < min:
    #         plt.savefig(os.path.join(os.getcwd(), 'data', '0618', exp_name, 'Epoch' + str(epoch) + '______.png'))
    #     else:
    #         plt.savefig(os.path.join(os.getcwd(), 'data', '0618', exp_name, 'Epoch' + str(epoch) + '.png'))
    #     plt.close()
    return test_loss_mse, test_loss_mae

@torch.no_grad()
def part_test(graph, data, N, T, epoch, exp_name, optimizer, test_idx, min, train_T):
    # test_num = int(T*test_ratio)-1
    # train_num = T-test_num

    test_num = T-train_T                                    # part test
    preds = torch.zeros([N, test_num]).cuda()
    cur_pred = data[test_idx, :, train_T:train_T+1].cuda()
    for t in range(test_num):
        cur_pred = graph.forward(cur_pred)
        # if torch.sum(torch.isnan(cur_pred)) > 0:
        #     print('TEST_nan')
        #     break
        preds[:, t] = cur_pred.squeeze(1)  # epoch wise
    preds = preds.cpu()
    real_avg = torch.sum(data[test_idx], axis=0)/N
    prd_avg = torch.sum(preds, axis=0)/N
    test_loss_mse = ((preds - data[test_idx, :, train_T+1:]) ** 2).mean()
    test_loss_mae = torch.abs(preds - data[test_idx, :, train_T+1:]).mean()
    if test_loss_mse < min:
        time = np.arange(T)
        time_test = np.arange(test_num)+train_T+1
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Lr = " + str(optimizer.param_groups[0]['lr']) + 'MAE=' + str(test_loss_mae))
        plt.xlabel("Time t/T$_{max}$")
        plt.ylabel("Node state x$_i$(t)")
        plt.tick_params(axis="y", direction="in", )
        plt.tick_params(axis="x", direction="in", )
        plt.scatter(time, real_avg[1:], c='g', s=0.3)
        plt.scatter(time_test, prd_avg, c='r', s=0.3)
        # ave = np.mean(data, axis=1)
        # for i in range(N):
        #     if i % 30 == 7:
        #         plt.scatter(time, data[test_idx, i, 1:], c='k', s=0.1)
        #         # plt.scatter(time_test, preds[i], c='r', s=0.1)
        #         plt.scatter(time, preds[i], c='r', s=0.1)
        plt.subplot(1, 2, 2)
        # plt.title('Selected nodes test_idx=' + str(test_idx))
        plt.title('MSE=' + str(test_loss_mse.detach().item()))
        plt.xlabel("Time t/T$_{max}$")
        plt.ylabel("Node state x$_i$(t)")
        plt.tick_params(axis="y", direction="in", )
        plt.tick_params(axis="x", direction="in", )
        # list = [0, 1, 3, 6, 8, 18, 33, 155, 159, 189, 239, 324]  # high-school
        list = [0, 3, 11, 20, 48, 76, 90, 110, 130]                                 # primary-school
        # list = [0, 3, 11, 20, 48, 76, 5, 90, 120, 150, 180]
        for i in list:
            plt.scatter(time, data[test_idx, i, 1:], c='k', s=0.5)
            plt.scatter(time_test, preds[i], c='r', s=0.5)
        # plt.show()
        plt.savefig(os.path.join(os.getcwd(), 'data', '0618', exp_name, 'Epoch' + str(epoch) + '.png'))
        plt.close()
    return test_loss_mse, test_loss_mae


def train():

    N = 327  # real edge number E = 7937              #  high school
    E = 7937
    beta = 0.1      # high school
    T = 1500
    T_max = 1.5
    # seq_len = 248          # 1 input -> 9 output
    initial_coef = 0  # para_matrix initialization
    # lamda = 2
    curing = 1

    # train_data_ratio = 0.5
    # test_data_ratio = 1 - train_data_ratio
    # train_T = int(T*train_data_ratio)

    co_data = []
    ntrain = 1                 # ntrain
    min_mse = 1
    min_mae = 1

    lr = 0.001 
    batch_size = 1
    test_batch = 1
    k = 75                       # error-accumulated steps
    loss_coef1 = 1
    flag = 1
    torch.cuda.set_device(0)

     exp_name = 'Adam_evolving_N'+str(N)+'_E'+str(E)+'_k'+str(k)+'_coef2_'+str(loss_coef2)+'_ntrain' + str(ntrain)+'_' \
    print(exp_name)

    os.makedirs(os.path.join(os.getcwd(), 'data', '0817', exp_name), exist_ok=True)

     for n in tqdm(range(ntrain+1)):          # generate 2, 1 for train
         co_data.append(get_data_product(
             N=N, E=E, beta=beta, T_max=T_max, n_real=T + 1, curing=curing, lamda=lamda, plot_data=1, real_graph=0,edge_size=edge_size))
    get_data_product(N=N, E=E, beta=beta, T_max=T_max, n_real=T + 1, curing=curing, plot_data=1, real_graph=1,edge_size=edge_size)
    co_data = torch.stack([co_data[0], co_data[1]], dim=0)

    data_len = co_data[0].shape[1] - 1
    # part train
    # train_data, test_data = torch.split(co_data, (train_T + 1, T-train_T), dim=2)
    # train_len = train_T

    # full_train
    train_len = data_len
    train_data = co_data

    edge_list1 = get_L(N=N, E=E, lamda=2, real_graph=1, max_size=5)
    graph = HyperGraph(N, E, beta=beta, curing=curing, delta_t=T_max / T, edge_dist=edge_list1).cuda()
    graph.cuda()

    max_iteration = 750
    optimizer = torch.optim.Adam(graph.parameters(), lr=lr, eps=1e-08, weight_decay=0)
    # optimizer = torch.optim.SGD(graph.parameters(), lr=lr, momentum=0.99, weight_decay=1e-4, nesterov=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=5, threshold=0.0001,
    #     threshold_mode='abs', cooldown=0, min_lr=0, eps=1e-08, verbose=1)

    batch_idx_list = iter(torch.randperm(int(train_len - k)))

    # batch_idx_list = iter(torch.arange(train_len-k))

    def iter_():
        nonlocal batch_idx_list

        try:
            return next(batch_idx_list)
        except StopIteration:
            batch_idx_list = iter(torch.randperm(train_len - k))
            print('Epoch over')
            return next(batch_idx_list)

    # torch.set_anomaly_enabled(True)

    for i in tqdm(list(range(max_iteration))):
        cur_batch_idx = [iter_() for _ in range(batch_size)]
        # loss_sum = 0
        for c in range(ntrain):
            cur_train = torch.stack([train_data[c, :, _start_idx: _start_idx + k + 1].cuda()
                                     for _start_idx in cur_batch_idx], dim=0)  # [batch_size, N, k+1]
            for b in range(batch_size):
                nan = 0
                loss = train_in_batch(cur_train=cur_train, graph=graph, k=k, b=b,
                                           lc1=loss_coef1, lc2=loss_coef2, c=c)
                # print(f"Loss={loss.detach().item()}")
                # loss = loss/(ntrain*3)         # 12.13 16:18
                loss = loss / (ntrain)
                loss.backward()

        print(f"Train Loss : {loss.detach().item()}")
        optimizer.step()
        # for col in torch.nonzero(torch.sum(graph.para_matrix, axis=0) > graph.N_node*2):
        #     graph.para_matrix.detach()[:, col] = torch.zeros(N).unsqueeze(1).cuda()
        #     print('zero')
        optimizer.zero_grad()
        if torch.sum(torch.isnan(graph.para_matrix)) > 0:
            print("matrix_nan")
        if i % test_batch == 0:
            mse, mae = test(graph=graph, data=co_data, N=N, T=T, epoch=i, exp_name=exp_name, optimizer=optimizer,
                             test_idx=ntrain, min=min_mse)
            # mse, mae = part_test(graph=graph, data=co_data, N=N, T=T, epoch=i, exp_name=exp_name, optimizer=optimizer,
            #                 test_idx=0, min=min_mse,  train_T=train_T)
            # if (test_loss < 0.001) and flag:
            #     optimizer.param_groups[0]['lr'] /= 2
            #     flag = 0
            if mse < min_mse:
                min_mse = mse
                min_mae = mae
                torch.save(graph.para_matrix.detach(),
                           os.path.join(os.getcwd(), 'data', '0618', exp_name, 'Best_matrix'))
                print(f"_____________________________________________________________Min mse at {i} Epoch: {mse.detach().item()}")
                print(f"_____________________________________________________________mae at {i} Epoch: {mae.detach().item()}")
            else:
                if i % (test_batch*20) == 0:
                    print(f"________________________________Test Loss at {i} Epoch: {mse.detach().item()}")

    print(f"Min mse: {min_mse.detach().item()}")
    print(f"Min mae: {min_mae.detach().item()}")
    print(exp_name)


train()

# for e in tqdm(list(range(total_epoch)), desc="Epoch"):
#     # active_train_len = train_len - seq_len + 1
#     # batch_bar = tqdm(list(range( // batch_size)), desc="Batch")
#     # batch_bar = range(active_train_len // batch_size)
#     # batch_idx_list = torch.randperm(active_train_len)       # random batch
#           #
#     # torch.save(batch_idx_list, './data/batch_idx_list.npy')
#     # batch_idx_list = torch.load('./data/batch_idx_list.npy')
#     for b in batch_bar:
#         cur_batch_idx = batch_idx_list[b * batch_size: (b + 1) * batch_size]
#         czur_train_data = [train_data[:, _start_idx: _start_idx + seq_len].cuda() for _start_idx in cur_batch_idx]
#         pred_loss = torch.zeros(1, device=cur_train_data[0].device)
#         for sample_sequence in cur_train_data:
#             cur_sequence_pred = []
#             cur_sample = sample_sequence[:, 0:1]
#             for _ in range(seq_len - 1):
#                 cur_sample = graph.forward(cur_sample)
#                 cur_sequence_pred.append(cur_sample)
#             cur_sequence_pred = torch.cat(cur_sequence_pred, dim=1)
#             cur_pred_loss = graph.predict_loss(cur_sequence_pred, sample_sequence[:, 1:])
#             pred_loss += cur_pred_loss      # weights?
#         pred_loss = pred_loss / batch_size
#         # hyper_count_loss = graph.hyper_count_loss()
#         # loss = hyper_count_loss + pred_loss * pred_loss_coef
#         # print(f"Train Loss at {e} Epoch = : {pred_loss.detach().item()}")
#         # batch_bar.set_postfix({
#         # "loss": loss.item(),
#         # "pred_loss": pred_loss.detach().item()},
#         # "hyper_count_loss": hyper_count_loss.detach().item()},
#         # refresh=True)
#
#         pred_loss.backward()
#         optimizer.step()
#         # graph.para_matrix.data = torch.sigmoid(graph.para_matrix)
