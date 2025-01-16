import random
import networkx as nx
import matplotlib.pyplot as plt

def init_graph(v_0):
    G = nx.Graph()
    for v_start in range(v_0):
        for v_stop in range(v_start + 1, v_0):
            G.add_edge(v_start, v_stop)
    return G

def add_new_node(G, new_edges_per_node):
    num_v_now = G.number_of_nodes()
    d_sum = sum(d for _, d in G.degree())
    d_dict = dict(G.degree())
    prob_choose = [d / d_sum for d in d_dict.values()]
    list_choose = list(range(num_v_now))
    choose_already = []

    while len(choose_already) < new_edges_per_node:
        v_stop = random.choices(population=list_choose, weights=prob_choose, k=1)[0]
        if v_stop in choose_already:
            continue
        choose_already.append(v_stop)
        G.add_edge(num_v_now, v_stop)
        d_sum += 2   
        prob_choose[v_stop] += 1 / d_sum   


def network_model(num_v, v_0, new_edges_per_node):
    G = init_graph(v_0)
    for _ in range(v_0, num_v):
        add_new_node(G, new_edges_per_node)
    return G


if __name__ == '__main__':
    G_BA1 = BA_model(1000, 5, 5)
    edgesBA1 = nx.edges(G_BA1)
    data = [' '.join(map(str, x)) for x in edgesBA1]
    with open('data1.txt', 'w', encoding='utf-8') as f:
        f.writelines([i + '\n' for i in data])

    G_BA2 = network_model(1000, 5, 4)
    edgesBA2 = nx.edges(G_BA2)
    data = [' '.join(map(str, x)) for x in edgesBA2]
    with open('data2.txt', 'w', encoding='utf-8') as f:
        f.writelines([i + '\n' for i in data])

    # 可视化
    y = nx.degree_histogram(G_BA2)
    plt.loglog(y)
    plt.show()