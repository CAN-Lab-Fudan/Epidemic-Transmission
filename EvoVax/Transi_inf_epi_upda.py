import math
import networkx as nx

def generationNetwork(N):
    BA1 = nx.barabasi_albert_graph(N, 3)
    BA2 = nx.barabasi_albert_graph(N, 3)
    return BA1, BA2

def adjacency_matrices(BA1, BA2):
    A = np.array(nx.adjacency_matrix(BA1).todense())
    B = np.array(nx.adjacency_matrix(BA2).todense())
    return A, B

def initialize_nodes(N):
    Node = [i for i in range(N)]
    ran = [random.random() for i in range(N)]
    US = []; AS = []; AV = []; US_AI = []; AS_AI = []; AV_AI = []; US_AI_AR = []; AS_AI_AR = []; AV_AI_AR = []

    selectNode = random.sample(Node, 5)  
    for i in selectNode:
        US_AI.append(i)
    for i in US_AI:
        Node.remove(i)

    for i in Node:
        if ran[i] < 0.33:
            US.append(i)
        elif ran[i] < 0.66:
            AS.append(i)
        else:
            AV.append(i)

    return US, AS, AV, US_AI, AS_AI, AV_AI, US_AI_AR, AS_AI_AR, AV_AI_AR

def feimiRule(U1, U2, kk):
    return 1 / (1 + math.exp(-kk * (U1 - U2)))

def update_strategies(N, US, AS, AV, US_AI_AR, AS_AI_AR, AV_AI_AR, BA2, P_dict):
    for i in range(N):
        if i in US:
            update_node_strategy(i, US, AV, AV_AI_AR, BA2, P_dict['P_US_AV'], P_dict['P_US__AV_AI'])
        elif i in AS:
            update_node_strategy(i, AS, AV, AV_AI_AR, BA2, P_dict['P_AS_AV'], P_dict['P_AS__AV_AI'])
        elif i in AV:
            update_node_strategy(i, AV, US, US_AI_AR, BA2, P_dict['P_AV_US'], P_dict['P_AV__US_AI'])
            update_node_strategy(i, AV, AS, AS_AI_AR, BA2, P_dict['P_AV_AS'], P_dict['P_AV__AS_AI'])
        elif i in US_AI_AR:
            update_node_strategy(i, US_AI_AR, AV, AV_AI_AR, BA2, P_dict['P_US_AI__AV'], P_dict['P_US_AI__AV_AI'])
        elif i in AS_AI_AR:
            update_node_strategy(i, AS_AI_AR, AV, AV_AI_AR, BA2, P_dict['P_AS_AI__AV'], P_dict['P_AS_AI__AV_AI'])
        elif i in AV_AI_AR:
            update_node_strategy(i, AV_AI_AR, US, US_AI_AR, BA2, P_dict['P_AV_AI__US'], P_dict['P_AV_AI__US_AI'])
            update_node_strategy(i, AV_AI_AR, AS, AS_AI_AR, BA2, P_dict['P_AV_AI__AS'], P_dict['P_AV_AI__AS_AI'])
    return US, AS, AV, US_AI_AR, AS_AI_AR, AV_AI_AR

def update_node_strategy(i, current_group, other_group, other_AI_group, BA2, P_current_other, P_current_other_AI):
    neiNodeS = list(nx.neighbors(BA2, i))
    ranNode = random.choice(neiNodeS)

    if ranNode in other_group:
        if random.random() < P_current_other:
            current_group.remove(i)
            other_group.append(i)
    elif ranNode in other_AI_group:
        if random.random() < P_current_other_AI:
            current_group.remove(i)
            other_group.append(i)
    else:
        current_group.remove(i)
        current_group.append(i)

def calculate_probabilities(pAV, pUS, pAS, pUS_AI, pAS_AI, pAV_AI, Cr, rr, kk):
    U_US_WE = 0
    U_AS_WE = 0
    U_AV_WE = update.weightEffect(pAV, rr) * (-Cr)
    U_AV_AI_WE = update.weightEffect(pAV_AI, rr) * (-Cr - 1)
    U_US_AI_WE = update.weightEffect(pUS_AI, rr) * (-1)
    U_AS_AI_WE = update.weightEffect(pAS_AI, rr) * (-1)

    P_dict = {
        'P_US_AV': update.feimiRule(U_US_WE, U_AV_WE, kk),
        'P_US_AI__AV': update.feimiRule(U_US_AI_WE, U_AV_WE, kk),
        'P_US__AV_AI': update.feimiRule(U_US_WE, U_AV_AI_WE, kk),
        'P_US_AI__AV_AI': update.feimiRule(U_US_AI_WE, U_AV_AI_WE, kk),

        'P_AS_AV': update.feimiRule(U_AS_WE, U_AV_WE, kk),
        'P_AS_AI__AV': update.feimiRule(U_AS_AI_WE, U_AV_WE, kk),
        'P_AS__AV_AI': update.feimiRule(U_AS_WE, U_AV_AI_WE, kk),
        'P_AS_AI__AV_AI': update.feimiRule(U_AS_AI_WE, U_AV_AI_WE, kk),

        'P_AV_US': update.feimiRule(U_AV_WE, U_US_WE, kk),
        'P_AV_AS': update.feimiRule(U_AV_WE, U_AS_WE, kk),
        'P_AV__US_AI': update.feimiRule(U_AV_WE, U_US_AI_WE, kk),
        'P_AV__AS_AI': update.feimiRule(U_AV_WE, U_AS_AI_WE, kk),
        'P_AV_AI__US': update.feimiRule(U_AV_AI_WE, U_US_WE, kk),
        'P_AV_AI__AS': update.feimiRule(U_AV_AI_WE, U_AS_WE, kk),
        'P_AV_AI__US_AI': update.feimiRule(U_AV_AI_WE, U_US_AI_WE, kk),
        'P_AV_AI__AS_AI': update.feimiRule(U_AV_AI_WE, U_AS_AI_WE, kk),
    }
    return P_dict

import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
import updateStrategies as update

def main():
    N = 1000
    startTime = datetime.datetime.now()

    BA1, BA2 = update.generationNetwork(N)
    A, B = adjacency_matrices(BA1, BA2)

    US, AS, AV, US_AI, AS_AI, AV_AI, US_AI_AR, AS_AI_AR, AV_AI_AR = initialize_nodes(N)

    endPV_mean = [0] * 10
    endAR_mean = [0] * 10

    for c in range(10):
        Cr = np.linspace(0, 1, 10)
        listV = []
        listAR = []
        # ******
        endPV_mean[c] = np.mean(listV)
        endAR_mean[c] = np.mean(listAR)

    np.savetxt('result_x.txt', endPV_mean, fmt='%.10f')
    np.savetxt('result_I.txt', endAR_mean, fmt='%.10f')

    plt.plot(np.array(range(10)) / 10, endPV_mean, marker='*', mec='g', mfc='w')
    plt.xlabel(r'$C_r$')
    plt.ylabel(r'$\rho^V$')
    plt.show()

    plt.plot(np.array(range(10)) / 10, endAR_mean, marker='*', mec='g', mfc='w')
    plt.xlabel(r'$C_r$')
    plt.ylabel(r'$\rho^I$')
    plt.show()

    endTime = datetime.datetime.now()
    print(endTime - startTime)


if __name__ == "__main__":
    main()
