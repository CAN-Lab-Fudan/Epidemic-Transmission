import numpy as np
import networkx as nx
import heapq
import matplotlib.pyplot as plt
from tqdm import tqdm

def run_gillespie_SIS(G, beta, delta, epsilon, max_time):
    N = G.number_of_nodes()
    state = np.zeros(N, dtype=int)
    event_queue = []
    heapq.heapify(event_queue)

    current_time = 0

    def schedule_recovery(node, current_time):
        recovery_time = current_time + np.random.exponential(1 / delta)
        heapq.heappush(event_queue, (recovery_time, 'recovery', node, None))

    def schedule_infection(node, neighbor, current_time):
        infection_time_neighbor = current_time + np.random.exponential(1 / beta)
        recovery_time_node = [t for t, et, n, neighbor in event_queue if n == node and et == 'recovery'][0]
        if infection_time_neighbor < recovery_time_node:
            heapq.heappush(event_queue, (infection_time_neighbor, 'infection', node, neighbor))

    def schedule_spontaneous_infection(current_time):
        event_time = current_time + np.random.exponential(1 / (epsilon * N))
        node = np.random.randint(0, N)
        heapq.heappush(event_queue, (event_time, 'spontaneous_infection', node, None))

    def create_spreading_events(node, current_time):
        for neighbor in G.neighbors(node):
            schedule_infection(node, neighbor, current_time)

    schedule_spontaneous_infection(current_time=0)

    infection_count = []
    time_points = []

    with tqdm(total=max_time) as pbar:
        while current_time < max_time and event_queue:
            time, event_type, node, neighbor = heapq.heappop(event_queue)
            pbar.update(time - current_time)
            current_time = time

            if event_type == 'recovery' and state[node] == 1:
                state[node] = 0
            elif event_type == 'infection' and state[neighbor] == 0:
                if G.has_edge(node, neighbor):
                    state[neighbor] = 1
                    schedule_recovery(neighbor, current_time)
                    create_spreading_events(neighbor, current_time)
                    if state[node] == 1:    
                        schedule_infection(node, neighbor, current_time)
            elif event_type == 'spontaneous_infection':
                if state[node] == 0:
                    state[node] = 1
                    schedule_recovery(node, current_time)
                    create_spreading_events(node, current_time)
                schedule_spontaneous_infection(current_time)

            infection_count.append(np.sum(state))
            time_points.append(current_time)

    return time_points, infection_count

def plot_infection_over_time(G, beta, delta, epsilon, max_time, title):
    time_points, infection_proportion = run_gillespie_SIS(G, beta, delta, epsilon, max_time)
    plt.plot(time_points, infection_proportion, label=title)
    plt.xlabel('time', fontsize=15)
    plt.ylabel('number of infected nodes', fontsize=15)
    plt.title('Infection count Over Time' )
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)
    plt.legend()

delta = 1 
epsilon = 0.001  
max_time = 30  
beta = 0.6 * delta

N = 500
p = 2 * np.log(N) / N
G = nx.erdos_renyi_graph(N, p)

plt.figure(figsize=(12, 8))
plot_infection_over_time(G, beta, delta, epsilon, max_time, "ER Graph")
plt.show()