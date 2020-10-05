import torch
import numpy as np
import networkx as nx
from utils import gumbel_softmax_3d # for batch version sampling
from torch.nn.functional import gumbel_softmax # for single samping
from scipy.io import mmread
import matplotlib.pyplot as plt
# %matplotlib inline

class Modularity():
    def __init__(self, G, seed=0, device='cuda:0'):
        self.G = G
        self.seed = seed
        self.device = device
        if self.seed > 0:
            torch.manual_seed(self.seed)
        self.nnodes = G.number_of_nodes()
        self.nedges = G.number_of_edges()
        self.mod_mat = nx.modularity_matrix(G).astype(np.float32)
        
    def gumbel_optimization(self, ncoms, bs=256, max_iters=2000, lr=1e-2, init_tau=0.5, final_tau=0.1, diff=1e-8):
        """drop temperature from 20 to 0.1 in 2000 steps. You can choose to fix tau"""
        n = self.nnodes
        device = self.device
        B = torch.tensor(self.mod_mat, device=device)
        # we choose to directly optimize logits to avoid softmax
        x = -1 * torch.rand(bs, n, ncoms, device=device)
        x.requires_grad = True # x is learnable
        optimizer = torch.optim.Adam([x], lr=lr)
        tau = init_tau
        decay = (init_tau - final_tau) / max_iters
        Q_best = torch.ones(1, device=device)
        Q_best_idx = 0
        for i in range(max_iters):
            Q_old = Q_best.clone()
            optimizer.zero_grad()
            s = gumbel_softmax_3d(x, tau=tau, hard=True) # [bs, n, ncoms]
            mod_list = [torch.trace(s[idx, :, :].t() @ B @ s[idx, :, :]) for idx in range(bs)]
            mod_tensor = torch.stack(mod_list) # convert list to torch.tensor
            loss = -1 * torch.mean(mod_tensor)
            loss.backward()
            optimizer.step()
            tau -= decay # drop temperature
            with torch.no_grad():
                s = gumbel_softmax_3d(x, tau=tau, hard=True) # [bs, n, ncoms]
                mod_list = [torch.trace(s[j, :, :].t() @ B @ s[j, :, :]) for j in range(bs)]
                mod_tensor = torch.stack(mod_list) # we will select the best result from this list
                Q = torch.max(mod_tensor)
                idx = torch.argmax(mod_tensor)
                if Q > Q_best:
                    Q_best = Q
                    Q_best_idx = idx
                if torch.abs(Q - Q_old) < diff:
                    break
#                 print('Current best result: %.5f' % (Q_best.cpu().numpy()/ (2 * self.nedges)))
        Q_best = Q_best.cpu().numpy() / (2 * self.nedges)
        best_partition = s[Q_best_idx, :, :].cpu().numpy() # select the best partion
        # print('Best Q value with %i communities: %.5f' % (ncoms, Q_best))
        return Q_best, best_partition

###########################################################################
# Zachary
# print('\nZachary\n')
# G = nx.karate_club_graph()
# mod = Modularity(G)
# for num_com in range(2, 6):
#     results = []
#     partitions = []
#     for _ in range(100):
#         Q, partition = mod.gumbel_optimization(num_com)
#         results.append(Q)
#         partitions.append(partition.sum(0))
#     Q_max = max(results)
#     index = results.index(Q_max)
#     print('Number of %i with best results: %.5f' % (num_com, Q_max))
#     print('Partition checking:' + str(partitions[index]))
#     print('\n')
###########################################################################

###########################################################################
# Jazz
# load data
# print('\nJazz\n')
# data = mmread('./data/jazz.mtx')
# F = nx.from_scipy_sparse_matrix(data)
# G = nx.Graph()
# for edge in F.edges():
#     G.add_edges_from([edge])
# mod = Modularity(G)

# for num_com in range(3, 8):
#     results = []
#     partitions = []
#     for _ in range(100):
#         Q, partition = mod.gumbel_optimization(num_com)
#         results.append(Q)
#         partitions.append(partition.sum(0))
#     Q_max = max(results)
#     index = results.index(Q_max)
#     print('Number of %i with best results: %.5f' % (num_com, Q_max))
#     print('Partition checking:' + str(partitions[index]))
#     print('\n')
###########################################################################

###########################################################################
# Email
# print('\nEmail\n')
# # laod data
# def load_graph(filename):
#     G = nx.Graph() # undirected graphs
#     for line in open(filename):
#         strlist = line.split(' ')
#         n1 = int(strlist[0])
#         n2 = int(strlist[1])
# #         weight = float(strlist[w_index])
# #         G.add_weighted_edges_from([(n1, n2, weight)])
#         G.add_edges_from([(n1, n2)])
#     return G

# G = load_graph('./data/email-univ.edges')
# mod = Modularity(G)

# for num_com in range(7, 16):
#     results = []
#     partitions = []
#     for _ in range(100):
#         Q, partition = mod.gumbel_optimization(num_com)
#         results.append(Q)
#         partitions.append(partition.sum(0))
#     Q_max = max(results)
#     index = results.index(Q_max)
#     print('Number of %i with best results: %.5f' % (num_com, Q_max))
#     print('Partition checking:' + str(partitions[index]))
#     print('\n')
###########################################################################

###########################################################################
# bio
print('\nBio\n')
# laod data
def load_graph(filename):
    G = nx.Graph() # undirected graphs
    for line in open(filename):
        strlist = line.split(' ')
        n1 = int(strlist[0])
        n2 = int(strlist[1])
#         weight = float(strlist[w_index])
#         G.add_weighted_edges_from([(n1, n2, weight)])
        G.add_edges_from([(n1, n2)])
    return G

G = load_graph('./data/bio-celegans.edges')
mod = Modularity(G)

for num_com in range(6, 18):
    results = []
    partitions = []
    for _ in range(100):
        Q, partition = mod.gumbel_optimization(num_com)
        results.append(Q)
        partitions.append(partition.sum(0))
    Q_max = max(results)
    index = results.index(Q_max)
    print('Number of %i with best results: %.5f' % (num_com, Q_max))
    print('Partition checking:' + str(partitions[index]))
    print('\n')
###########################################################################

