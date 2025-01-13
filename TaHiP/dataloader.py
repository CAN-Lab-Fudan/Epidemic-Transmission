import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os


import seaborn as sns
import pandas as pd


def real_graph_loader():
    hyperedge_size = []
    hyperedges = []
    nodes = []
    max_node = 0
    N = 1006
    # hyperdegree = np.zeros(N)

    for line in open('./data/Realworld_hypergraph/email-Eu/email-Eu-nverts.txt', 'r'):
    # # for line in open('./data/Realworld_hypergraph/contact-high-school/contact-high-school-nverts.txt', 'r'):
        hyperedge_size.append(int(line))
    
    for line in open('./data/Realworld_hypergraph/email-Eu/email-Eu-simplices.txt', 'r'):
    # # for line in open('./data/Realworld_hypergraph/tags-math/tags-math-sx-simplices.txt', 'r'):
        nodes.append(int(line))
        if int(line) > max_node:
            max_node = int(line)


    # for line in open('./data/Realworld_hypergraph/senate-com/hyperedges-senate-committees.txt', 'r'):
    # for line in open('./data/Realworld_hypergraph/senate-bills/hyperedges-senate-bills.txt', 'r'):
    # for line in open('./data/Realworld_hypergraph/house-bills/house-bills/hyperedges-house-bills.txt', 'r'):
    # for line in open('./data/Realworld_hypergraph/blues/hyperedges.txt', 'r'):
    #     temp = line.split(',')
    #     edge = set()
    #     for node in temp:
    #         edge.add(int(node))
    #         if int(node) > max_node:
    #             max_node = int(node)
    #     if edge not in hyperedges:
    #         hyperedges.append(edge)
            # hyperedge_size.append(len(edge))
            # for node in edge:
            #     hyperdegree[int(node) - 1] += 1  # hyperdegree[0] is the degree of node 1
    #
    # edge_size = [len(x) for x in hyperedges]
    #
    # overlapness = sum(hyperdegree) / len(hyperdegree)
    # density = max_node/len(edge_size)
    # avg_edge_size = sum(edge_size)/len(edge_size)
    # max_size = max(edge_size)
    # per = edge_size.count(2)/len(edge_size)

    #
    unique_edge = []
    flag = 0
    for size in hyperedge_size:
        hyperedge = set()
        for i in range(size):
            hyperedge.add(nodes[flag+i])
        flag += size
        if hyperedge not in hyperedges:
            hyperedges.append(hyperedge)
            unique_edge.append(len(hyperedge))
            # for node in hyperedge:
            #     hyperdegree[node-1] += 1        # hyperdegree[0] is the degree of node 1

    # torch.save(hyperdegree, './data/Realworld_hypergraph/hypergraph/email-Eu')
    torch.save(hyperedges, './data/Realworld_hypergraph/hypergraph/email-Eu/email-Eu-edges')
    # torch.save(hyperedges, './data/Realworld_hypergraph/hypergraph/house-bills/house-bills-edges')
    # torch.save(hyperedges, './data/Realworld_hypergraph/hypergraph/senate-bills/senate-bills-edges')
    # torch.save(hyperdegree, './data/0304/surrogate matrix/Hbills-hyperdegree')
    # torch.save(edge_size, './data/0304/surrogate matrix/Hbills-hyperedge-size')
    # per = unique_edge.count(2)/len(unique_edge)

    # hyperdegree.sort()

