import argparse
import math
import time
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.functional import gumbel_softmax
from utils import gumbel_softmax_3d

# 梯度优化算法类
class my_parameters(nn.Module):
    
    def __init__(self, batch_size=128, n=1024, device='cuda:0'):    
        super(my_parameters, self).__init__()
        self.batch_size = batch_size
        self.n = n
        rand = torch.randn(batch_size, n, 1, device=device) * 1e-5
        self.init_value = rand
        self.pps = Parameter(self.init_value)
        
    def expand_tensor(self, x):
        return x.unsqueeze(0).repeat(self.batch_size, 1)
        
class SKModel():
    def __init__(self, n=20, seed=0, device='cuda:0'):
        self.n = n
        self.seed = seed
        self.device = device
        if self.seed > 0:
            torch.manual_seed(self.seed)
        self.J = torch.randn(self.n, self.n) / math.sqrt(self.n)
        self.J = torch.triu(self.J, diagonal=1)
        self.J = self.J + self.J.t()
        self.J_np = self.J.numpy()

    def gumbel_optimization(self, bs=128, max_iters=20000, lr=1, eta=1e-3, init_tau=20, final_tau=1, T=100,u=8):
        n = self.n
        device = self.device
        J = self.J.to(device)
        best_log = []
        # learnable parameters
        # x = torch.randn(bs, n, 1, device=device, requires_grad=True)
        nnn = my_parameters(bs, n, device=device)
        #x = torch.randn(bs, n, 1, device=device) * 1e-5
        #x.requires_grad = True
        optimizer = optim.Adam(nnn.parameters(), lr=lr)
        tau = init_tau
        diff=1e-8
        decay = (init_tau - final_tau) / max_iters
        E_best = torch.ones(1, device=device)

        for i in range(max_iters):
            E_old = E_best.clone()
            optimizer.zero_grad()
            probs = torch.empty(bs, n, 2, device=device)
            p = torch.sigmoid(nnn.pps)
            probs[:, :, 0] = p.squeeze()
            probs[:, :, -1] = 1 - probs[:, :, 0]
            logits = torch.log(probs + 1e-10)
            s = 2 * gumbel_softmax_3d(logits, tau=tau, hard=False)[:, :, 0] - 1
            E = -0.5 * torch.sum((s @ J) * s, dim=1) / n
            constraint = torch.sum(nnn.pps ** 2) / n / bs
            loss = torch.mean(E) + eta * constraint
            loss.backward()
            optimizer.step()
            tau -= decay
            with torch.no_grad():
                #if i % 100 == 0:                    
                    #print(i,'Current best result: %.8f' % (E_best.cpu().numpy()))
                s = 2 * gumbel_softmax_3d(logits, tau=tau, hard=True)[:, :, 0] - 1
                E = -0.5 * torch.sum((s @ J) * s, dim=1) / n
                Emin = torch.min(E)
                Emean = torch.mean(E)
                if Emin < E_best:
                    E_best = Emin
                
                        
                # 演化算法
                if i % T == 0 and i > 0:
                    # 找到最好个体的下标
                    mindx = torch.argsort(E, dim=0,descending=False)
                    # 将种群按照适应度从小到大排序
                    maxdx = torch.argsort(E, dim=0,descending=True)
                    for j in range(bs//u):
                        #对1/8个体进行循环
                        temp=nnn.pps.data[mindx[j], :, 0]
                        if np.random.randn() < -3:
                            temp = torch.randn(n) * 1e-5
                        #找出其中一个最差的个体，用temp替换掉。temp可以是当前个体中最好的，也可以是一个随机个体
                        nnn.pps.data[maxdx[j], :, 0]=temp
                        #print(nnn.pps.data.size())
                        #print(temp.size())

                  
                
                if torch.abs(Emin - E_old) < diff:
                    break
        return E_best.cpu().numpy()
        
def main():
    # settings
    parser = argparse.ArgumentParser('EvoGSO optimizing SK model energy')
    parser.add_argument('--n', type=int, default=1024,
                        help='size (default: 1024)')
    parser.add_argument('--bs', type=int, default=128,
                        help='batch size (default: 128)')
    parser.add_argument('--max_iters', type=int, default=2000,
                        help='iterations (default: 50000)')
    parser.add_argument('--lr', type=float, default=1.,
                        help='learning rate (default: 1)')
    parser.add_argument('--eta', type=float, default=1e-3,
                        help='weight decay (default: 1e-3)')
    parser.add_argument('--init-tau', type=float, default=20.,
                        help='initial tau in Gumbel-softmax (default: 20)')
    parser.add_argument('--final-tau', type=float, default=1.,
                        help='final tau in Gumbel-softmax (default: 1)')
    parser.add_argument('--instances', type=int, default=10,
                        help='number of ensembles (default: 1024)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='cuda device (default: cuda:0)')
    parser.add_argument('--T', type=int, default=100,
                        help='evolution cycle (default: 100)')
    parser.add_argument('--u', type=int, default=8,
                        help='proportion (default: 8)')
    args = parser.parse_args()
    n = args.n  # size
    bs = args.bs # batch size
    max_iters = args.max_iters
    init_tau = args.init_tau
    final_tau = args.final_tau
    lr = args.lr
    eta = args.eta
    instances = args.instances
    device = args.device
    T = args.T
    u = args.u
    torch.manual_seed(2050)

    # training
    results_arr = []
    for _ in range(instances):
        sk = SKModel(n, device=device)
        energy = sk.gumbel_optimization(bs, max_iters, lr, eta, init_tau, final_tau,T,u)
        #print('# %i \t energy: %.5f' % (_, energy))
        results_arr.append(energy)

   # print mean energy and std
    data = np.array(results_arr)
    data_mean = data.mean()
    data_std = np.sqrt(np.var(data, ddof=1))
    data_sem = data_std / np.sqrt(instances)
    print('EvoGSO SK model N: %i, bs: %i, instances: %i, cycle: %i, proportion: %i, mean: %.5f, std: %.5f, sem: %.7f' %
          (n, bs, instances, T, u, data_mean, data_std, data_sem))


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('Running time: %.5f s \n' % (end - start))