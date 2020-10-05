import torch
import numpy as np
import networkx as nx
from utils import gumbel_softmax_3d # for batch version sampling
from torch.nn.functional import gumbel_softmax # for single samping
from scipy.io import mmread
import matplotlib.pyplot as plt
#%matplotlib inline

class Modularity():
    def __init__(self, G, seed=0, device='cuda:3'):
        self.G = G
        self.seed = seed
        self.device = device
        if self.seed > 0:
            torch.manual_seed(self.seed)
        self.nnodes = G.number_of_nodes()
        self.nedges = G.number_of_edges()
        self.mod_mat = nx.modularity_matrix(G).astype(np.float32)
        
    def gumbel_optimization(self, ncoms, bs=128, max_iters=2000, lr=1e-2, init_tau=1, final_tau=0.1, diff=1e-10):
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
                #print('modtensor:',mod_tensor)
                Q = torch.max(mod_tensor)
                idx = torch.argmax(mod_tensor)
                if Q > Q_best:
                    Q_best = Q
                    Q_best_idx = idx
                    
                #if i % 100 == 0:
                    #print('#',i,Q.cpu().numpy() / (2 * self.nedges))
                    
                ## 演化算法
                #if i % 100 == 0 and i < 2000:
                #    # 找到最好个体的下标
                #    maxdx = torch.argsort(mod_tensor, dim=0, descending=True)
                #    # 将种群按照适应度从小到大排序
                #    mindx = torch.argsort(mod_tensor, dim=0,descending=False)
                #    for j in range(bs//4):
                #        #对1/8个体进行循环
                #        temp=x.data[maxdx[j], :, :]
                #        #if np.random.rand()<1:
                #        #    temp = torch.randn(n) * 1e-5
                #        #找出其中一个最差的个体，用temp替换掉。temp可以是当前个体中最好的，也可以是一个随机个体
                #        x.data[mindx[j], :, :]=temp
                #        
                #            #交叉变异
                #if i % 100 == 0 and i >= 2000:
                #    
                #    ratio = 1/4
                #    r = int(ratio * bs)
                #    m = 2e-3
                #    
                #    maxdx = torch.argsort(mod_tensor, dim=0, descending=True).squeeze()
                #    mindx = torch.argsort(mod_tensor, dim=0,descending=False).squeeze()
                #    #print('maxdx:',maxdx)
                #    #print('mindx:',mindx)
                #    L1 = random.sample(list(mindx[0:r]), r)
                #    L2 = random.sample(list(maxdx[0:r]), r)
                #    #print('L1:',L1)
                #    #print('L2:',L2)
                #    #print(loss_[L1[0]])
                #    for j in range(r//2):
                #        #print(L1[i])
                #        #print('before:',nnn.pps.data[L1[i], : ,0])
                #        #print(L2[i])
                #        #print('father:',nnn.pps.data[L2[i], :, 0])
                #        #print(L2[-(i+1)])
                #        #print('mother:',nnn.pps.data[L2[-(i+1)], :, 0])
                #        rand = torch.rand(n).cuda()
                #        x.data[L1[i], : ,:] = torch.where(rand < 0.5, 
                #                                                x.data[L2[i], :, :], 
                #                                                x.data[L2[r-1-i], :, :])
                #        x.data[L1[r-1-i], :, :] = torch.where(rand < 0.5, 
                #                                                x.data[L2[r-1-i], :, :], 
                #                                                x.data[L2[i], :, :])
                #        #print('after:',nnn.pps.data[L1[i], : ,0])
                #        rand = torch.rand(n).cuda()
                #        x.data[L1[i], : ,:] = torch.where(rand < m, 
                #                                                x.data[L1[i], : ,:], 
                #                                                torch.rand_like(x.data[L1[i], : ,:]))
                #        #count = torch.where(rand<m,torch.ones_like(rand),torch.zeros_like(rand))
                #        #print('mutation_count:',torch.sum(count))
                    
                if torch.abs(Q - Q_old) < diff:
                    break
#                 print('Current best result: %.5f' % (Q_best.cpu().numpy()/ (2 * self.nedges)))
        Q_best = Q_best.cpu().numpy() / (2 * self.nedges)
        best_partition = s[Q_best_idx, :, :].cpu().numpy() # select the best partion
        #print('Best Q value with %i communities: %.5f' % (ncoms, Q_best))
        return Q_best, best_partition
        
        
        
G = nx.karate_club_graph()
mod = Modularity(G)
print('########## Karate ##########')
for num_com in range(2, 11):
    results = []
    partitions = []
    for _ in range(10):
        Q, partition = mod.gumbel_optimization(num_com)
        results.append(Q)
        partitions.append(partition.sum(0))
        print('#',_,'results: %.5f' % (Q))
        print('Partition checking:' + str(partition.sum(0)))
        print([np.argmax(one_hot)for one_hot in partition])
    Q_max = max(results)
    index = results.index(Q_max)
    print('Best Q value with ',num_com, 'communities:: %.5f' % (Q_max))
    print('Partition checking:' + str(partitions[index]))
    print('\n')
print('####################')
    

    
    
# load data
print('########## Jazz ##########')
data = mmread('./data/jazz.mtx')
F = nx.from_scipy_sparse_matrix(data)
G = nx.Graph()
for edge in F.edges():
    G.add_edges_from([edge])

mod = Modularity(G)

for num_com in range(2, 11):
    results = []
    partitions = []
    for _ in range(10):
        Q, partition = mod.gumbel_optimization(num_com)
        results.append(Q)
        partitions.append(partition.sum(0))
        print('#',_,'results: %.5f' % (Q))
        print('Partition checking:' + str(partition.sum(0)))
        #print([np.argmax(one_hot)for one_hot in partition])
    Q_max = max(results)
    index = results.index(Q_max)
    print('Best Q value with ',num_com, 'communities:: %.5f' % (Q_max))
    print('Partition checking:' + str(partitions[index]))
    print('\n')
print('####################')
    
    

# laod data
print('########## Celegen ##########')
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

for num_com in range(2,11):
    results = []
    partitions = []
    for _ in range(10):
        Q, partition = mod.gumbel_optimization(num_com)
        results.append(Q)
        partitions.append(partition.sum(0))
        print('#',_,'results: %.5f' % (Q))
        print('Partition checking:' + str(partition.sum(0)))
        #print([np.argmax(one_hot)for one_hot in partition])
    Q_max = max(results)
    index = results.index(Q_max)
    print('Best Q value with ',num_com, 'communities:: %.5f' % (Q_max))
    print('Partition checking:' + str(partitions[index]))
    print('\n')
print('####################')
    
    

# laod data
print('########## Email ##########')
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

G = load_graph('./data/email-univ.edges')

mod = Modularity(G)

for num_com in range(2, 11):
    results = []
    partitions = []
    for _ in range(10):
        Q, partition = mod.gumbel_optimization(num_com)
        results.append(Q)
        partitions.append(partition.sum(0))
        print('#',_,'results: %.5f' % (Q))
        print('Partition checking:' + str(partition.sum(0)))
        #print('Partition checking:' + str(partition))
        #print([np.argmax(one_hot)for one_hot in partition])
    Q_max = max(results)
    index = results.index(Q_max)
    print('Best Q value with ',num_com, 'communities:: %.5f' % (Q_max))
    print('Partition checking:' + str(partitions[index]))
    print('\n')