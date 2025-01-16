import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Function to calculate protection level based on m, lamda1, and b
# def protection_level(lamda1, m, b):
#     if b is None:
#         raise ValueError("Parameter 'b' cannot be None.")
#
#     if m == 0:
#         return 0
#     if b == 0 and m == 1:
#         return lamda1
#     if 0 < b < 1 and m > 1:
#         return lamda1 + (1 - lamda1) * (1 - math.exp(-b * (m - 1)))
#
#     return lamda1  # Default case for invalid b values

def protection_level(lamda1, m, b):
    if b is None:
        raise ValueError("Parameter 'b' cannot be None.")

    # Handle case when m == 0
    if m == 0:
        return 0

    # Case when b == 0 and m == 1
    if b == 0 or m == 1:
        return lamda1

    # Case when 0 < b < 1 and m > 1
    if 0 < b < 1 and m > 1:
        return lamda1 + (1 - lamda1) * (1 - math.exp(-b * (m - 1)))

# Load BA graph from a file
def load_ba_graph(file_name):
    G = nx.Graph()
    try:
        with open(file_name, encoding='utf-8') as file:
            for line in file:
                head, tail = map(int, line.split())
                G.add_edge(head, tail)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_name} not found.")
    return G

# Function to get node degrees from the adjacency matrix
def get_node_degrees(adj_matrix):
    return np.sum(adj_matrix, axis=0)
#
# Plot the protection levels as a function of m
def plot_protection_levels(Xdai, m_b03, m_b05, m_b07):
    plt.figure(figsize=(10, 6))
    plt.plot(Xdai, m_b03, label=r'$b=0.3$', color='c', marker='s', markersize=8, linewidth=2, linestyle='-.')
    plt.plot(Xdai, m_b05, label=r'$b=0.5$', color='y', marker='p', markersize=10, linewidth=2, linestyle='-.')
    plt.plot(Xdai, m_b07, label=r'$b=0.7$', color='m', marker='d', markersize=10, linewidth=2, linestyle='-.')

    plt.xlabel(r'$m$', fontsize=20)
    plt.ylabel(r'$\omega_{m}$', fontsize=20)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.show()


# Main execution logic
def main():
    # Parameters
    b03, b05, b07 = 0.3, 0.5, 0.7
    lamda1 = 0.1
    N = 100  # Number of nodes in the graph

    # Load graph data (ensure 'data.txt' is correctly formatted)
    BA1 = load_ba_graph('ER100.txt')
    BA2 = load_ba_graph('ER100.txt')  # Assuming the same file is used for BA2

    # Convert adjacency matrices to NumPy arrays for efficient operations
    A = np.array(nx.adjacency_matrix(BA1).todense())
    B = np.array(nx.adjacency_matrix(BA2).todense())

    # Get node degrees from adjacency matrix
    degrees = get_node_degrees(A)

    # Calculate protection levels for different m values
    m_b03, m_b05, m_b07 = [], [], []
    for m in range(1, 20):  # m ranges from 1 to 19 (inclusive)
        m_b03.append(protection_level(lamda1, m, b03))
        m_b05.append(protection_level(lamda1, m, b05))
        m_b07.append(protection_level(lamda1, m, b07))

    # X axis values for plotting (corresponding to m)
    X = list(range(1, 20))

    # Plot the results
    # plot_protection_levels(X, m_b03, m_b05, m_b07)
    with open('protection_levels.txt', 'w') as f:
        f.write("m\tb03\tb05\tb07\n")
        for i in range(len(X)):
            f.write(f"{X[i]}\t{m_b03[i]}\t{m_b05[i]}\t{m_b07[i]}\n")


if __name__ == '__main__':
    main()
