import math
import random
import networkx as nx
import numpy as np

def generationNetwork(N):
    generateBA1 = nx.barabasi_albert_graph(N, 4)
    generateBA2 = nx.barabasi_albert_graph(N, 3)  # BA网络   平均度为5

    return generateBA1, generateBA2

def informationReceiveProbability111(m, lamda1, b):
    lamda = 0
    if m == 0 :
        lamda = 0
    if b == 0 or m == 1:
        lamda = lamda1
    if 0 < b <= 1 and m > 1:
        lamda = lamda1 + (1 - lamda1) * (1 - math.exp(- b * (m - 1)))
        # lamda = lamda1 + (1 - lamda1) * ((b ** (1 - math.exp(- (m - 1)/k))))
    return lamda

def informationReceiveProbability22(m, lamda1, b):
    lamda = 0
    if m == 0 :
        lamda = 0.1
    if b == 0 or m == 1:
        lamda = 1 - lamda1
    if 0 < b <= 1 and m > 1:
        lamda = 1 - (lamda1 + (1 - lamda1) * (1 - math.exp(- b * (m - 1))))
    return lamda

def informationReceiveProbability(m, lamda1, b):
    # lamda = 0
    if m == 0:
        return 0
    if b == 0 or m == 1:
        return lamda1
    if 0 < b <= 1 and m > 1:
        return lamda1 + (1 - lamda1) * (1 - math.exp(- b * (m - 1)))

def informationReceiveProbability2(m, lamda1, b):
    lamda = 0
    if m == 0:
        lamda = 0
    if b == 1 and m >= 2:
        lamda = 0.6
    if 0 < b < 1 and m > 1:
        lamda = lamda1 + (1 - lamda1) * (1 - math.exp(- b * (m - 1)))
        if lamda > 0.6:
            lamda = 0.6
    return lamda

#权重函数，计算主观概率
def weightEffect(l,rr):
    if l > 0:
        w_l = math.exp(-(-math.log(l)) ** rr)
    else:
        w_l= 0
    return w_l

#费米规则，更新策略
def feimiRule(Ui, Uj, k):
    W_si_sj = 1 / (1 + math.exp(- (Uj - Ui) / k))
    return W_si_sj










