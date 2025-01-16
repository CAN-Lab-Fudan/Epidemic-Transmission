import math
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import updateStrategies as update
import datetime

startTime = datetime.datetime.now()

alpha = 0.4
alpha1 = 0.6
delta = 0.6
miu = 0.8
lamda1 = 0.2

generation = 200

N = 1000

BA1, BA2 = update.generationNetwork(N)
A = np.array(nx.adjacency_matrix(BA1).todense())
B = np.array(nx.adjacency_matrix(BA2).todense())

k1 = []
for i in range(N):
  k1.append(sum(A[:, i]))

k2 = []
for i in range(N):
  k2.append(sum(B[:, i]))

commNei = np.zeros((N, N))
for i in range(N):
    neiNodeI = list(nx.neighbors(BA1,i))
    for j in range(N):
        commNei[j,i] = len(list(nx.common_neighbors(BA1, i, j)))

terCr = 31

endPA_mean = [i for i in range(terCr)]
endPI_mean = [i for i in range(terCr)]

for c in range(terCr):
    beta_U = c / (terCr - 1)

    listPA = []
    listPI = []
    for T in range(generation):
        Node = [i for i in range(N)]
        ran = [random.random() for i in range(N)]

        US = []
        AS = []
        AV = []
        US_AI = []
        AS_AI = []
        AV_AI = []
        US_AI_AR = []
        AS_AI_AR = []
        AV_AI_AR = []

        selectNode = random.sample(Node, 5)
        for i in selectNode:
            US_AI.append(i)
        for ii in US_AI:
            Node.remove(ii)

        for i in Node:
            if ran[i] < 0.33:
                US.append(i)
            elif ran[i] < 0.66:
                AS.append(i)
            else:
                AV.append(i)

        MMCA = 1000
        for t in range(MMCA):
            # 第一步，更新信息层
            for i in range(N):
                if i in US:
                    neiJ = []
                    for j in range(N):
                        neiJ.append((k1[j] ** alpha) * A[j, i])
                    sumNeiI = sum(neiJ)

                    sumJ = 0
                    for j in range(N):
                        wji = neiJ[j] / sumNeiI

                        if j in (AS + AV + US_AI + AS_AI + AV_AI + US_AI_AR + AS_AI_AR + AV_AI_AR):
                            sumJ += wji * 1
                        else:
                            sumJ += wji * 0

                    lan = sumJ
                    m1 = 0
                    numNeiNode1 = k1[i]
                    for j in range(N):
                        if j in (US_AI + AS_AI + AV_AI + US_AI_AR + AS_AI_AR + AV_AI_AR) and A[j, i] == 1:
                            m1 += 1
                    receivePro = 1 - math.exp(- (m1 / numNeiNode1))

                    if random.random() < lan * receivePro:
                        US.remove(i)
                        AS.append(i)
                    continue

                # 有意识
                if i in AS:
                    if random.random() < delta:
                        AS.remove(i)
                        US.append(i)
                    continue

            #  物理层更新
            for i in range(N):
                if i in US_AI:
                    if random.random() < miu:
                        US_AI.remove(i)
                        US_AI_AR.append(i)
                    continue

                if i in AS_AI:
                    if random.random() < miu:
                        AS_AI.remove(i)
                        AS_AI_AR.append(i)
                    continue

                if i in AV_AI:
                    if random.random() < miu:
                        AV_AI.remove(i)
                        AV_AI_AR.append(i)
                    continue

                if i in AS:
                    m2 = 0
                    for j in range(N):
                        if j in (AS + AV + US_AI + AS_AI + AV_AI + US_AI_AR + AS_AI_AR + AV_AI_AR) and B[j, i] == 1:
                            m2 += 1
                    perc = 1 - update.informationReceiveProbability(m2, lamda1, alpha1)  # 病毒传播
                    for j in range(N):
                        if B[j, i] == 1 and j in (US_AI + AS_AI + AV_AI) and random.random() < perc * beta_U:
                            AS.remove(i)
                            AS_AI.append(i)
                            break
                    continue

                if i in US:
                    for j in range(N):
                        if B[j, i] == 1 and j in (US_AI + AS_AI + AV_AI) and random.random() < beta_U:
                            US.remove(i)
                            US_AI.append(i)
                            break
                    continue

                if i in AV:
                    m2 = 0
                    for j in range(N):
                        if j in (AS + AV + US_AI + AS_AI + AV_AI + US_AI_AR + AS_AI_AR + AV_AI_AR) and B[j, i] == 1:
                            m2 += 1
                    perc = 1 - update.informationReceiveProbability(m2, lamda1, alpha1)

                    for j in range(N):
                        if B[j, i] == 1 and j in (US_AI + AS_AI + AV_AI) and random.random() < perc * beta_U:
                            AV.remove(i)
                            AV_AI.append(i)
                            break
                    continue

            PI = US_AI + AS_AI + AV_AI
            if len(PI) == 0:
                break

        listPI.append(len(US_AI_AR + AS_AI_AR + AV_AI_AR) / N)
        listPA.append(len(AS + AV + US_AI_AR + AS_AI_AR + AV_AI_AR) / N)

    endPI_mean[c] = np.mean(listPI)
    endPA_mean[c] = np.mean(listPA)

endTime = datetime.datetime.now()
print(endTime - startTime)

np.savetxt('PA.txt', endPA_mean, fmt='%.10f')
np.savetxt('PI.txt', endPI_mean, fmt='%.10f')

X = np.array([i for i in range(terCr)]) / (terCr - 1)

plt.plot(X, endPI_mean, label=r'$\rho^R$', color='deepskyblue', marker='D', linewidth=2, linestyle='-')
plt.plot(X, endPA_mean, label=r'$\rho^A$', color='darkgoldenrod', marker='D', linewidth=2, linestyle='-')
plt.xlabel('\beta^U')
plt.ylabel(r'$rho$')
plt.legend()
plt.grid()
plt.show()















