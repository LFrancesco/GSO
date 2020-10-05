import argparse
import torch
import math
import numpy as np
import time
import pickle 
from torch.nn.functional import gumbel_softmax
from utils import gumbel_softmax_3d


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

    def gumbel_optimization(self, bs=128, max_iters=20000, lr=1, eta=1e-3, init_tau=20, final_tau=1, diff=1e-8):
        n = self.n
        device = self.device
        J = self.J.to(device)
        # learnable parameters
        # x = torch.randn(bs, n, 1, device=device, requires_grad=True)
        x = torch.randn(bs, n, 1, device=device) * 1e-5
        x.requires_grad = True
        optimizer = torch.optim.Adam([x], lr=lr)
        tau = init_tau
        decay = (init_tau - final_tau) / max_iters
        E_best = torch.ones(1, device=device)

        for i in range(max_iters):
            E_old = E_best.clone()
            optimizer.zero_grad()
            probs = torch.empty(bs, n, 2, device=device)
            p = torch.sigmoid(x)
            probs[:, :, 0] = p.squeeze()
            probs[:, :, -1] = 1 - probs[:, :, 0]
            logits = torch.log(probs + 1e-10)
            s = 2 * gumbel_softmax_3d(logits, tau=tau, hard=False)[:, :, 0] - 1
            E = -0.5 * torch.sum((s @ J) * s, dim=1) / n
            constraint = torch.sum(x ** 2) / n / bs
            loss = torch.mean(E) + eta * constraint
            loss.backward()
            optimizer.step()
            tau -= decay
            with torch.no_grad():
                #if i % (int(max_iters / 20)) == 0:                      
                    #print('Current best result: %.5f' % (E_best.cpu().numpy()))
                s = 2 * gumbel_softmax_3d(logits, tau=tau, hard=True)[:, :, 0] - 1
                E = -0.5 * torch.sum((s @ J) * s, dim=1) / n
                E = torch.min(E)
                if E < E_best:
                    E_best = E
                if torch.abs(E - E_old) < diff:
                    break
        return E_best.cpu().numpy()
                

def main():
    # settings
    parser = argparse.ArgumentParser('Gumbel-softmax optimizing SK model energy')
    parser.add_argument('--n', type=int, default=1024,
                        help='size (default: 1024)')
    parser.add_argument('--bs', type=int, default=128,
                        help='batch size (default: 128)')
    parser.add_argument('--max_iters', type=int, default=50000,
                        help='iterations (default: 50000)')
    parser.add_argument('--lr', type=float, default=1.,
                        help='learning rate (default: 1)')
    parser.add_argument('--eta', type=float, default=1e-3,
                        help='weight decay (default: 1e-3)')
    parser.add_argument('--init-tau', type=float, default=20.,
                        help='initial tau in Gumbel-softmax (default: 20)')
    parser.add_argument('--final-tau', type=float, default=1.,
                        help='final tau in Gumbel-softmax (default: 1)')
    parser.add_argument('--instances', type=int, default=100,
                        help='number of ensembles (default: 1024)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='cuda device (default: cuda:0)')
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
    torch.manual_seed(2050)

    # training
    results_arr = []
    for i in range(instances):
        sk = SKModel(n, device=device)
        energy = sk.gumbel_optimization(bs, max_iters, lr, eta, init_tau, final_tau)
        #print('# %i \t energy: %.5f' % (i, energy))
        results_arr.append(energy)

   # print mean energy and std
    data = np.array(results_arr)
    data_mean = data.mean()
    data_std = np.sqrt(np.var(data, ddof=1))
    data_sem = data_std / np.sqrt(instances)
    print('Batch Gumbel-softmax N: %i, bs: %i, instances: %i, mean: %.5f, std: %.5f, sem: %.7f' %
          (n, bs, instances, data_mean, data_std, data_sem))


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('Running time: %.5f s \n' % (end - start))
