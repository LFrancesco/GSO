import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
# %matplotlib inline
import torch
from torch.nn.functional import gumbel_softmax
import scipy
from torch.utils.data import DataLoader
import argparse

# def logistic_map(x, lambd=3.5):
#     return lambd * x * (1 - x)

# def one_step_dynamics(current_state, A, s=0.2, lambd=3.88, eps=0, device='cpu'):
#     """Given x_t (n by 1) and coupling A, produce x_t+1"""
#     diags = A.sum(1)
#     if isinstance(current_state, np.ndarray):
#         noise = np.random.randn(n, 1) * eps
#     else:
#         noise = torch.randn(n, 1) * eps
#         noise = noise.to(device)
#     next_state = (1 - s) * logistic_map(current_state, lambd=lambd) + s / diags * A @ logistic_map(current_state, lambd=lambd)
#     next_state += noise
#     return next_state

# def gumbel_sample(x, temp=1, hard=False):
#     logp = x.view(-1, 2)
#     out = gumbel_softmax(logp, temp, hard)
#     out = out[:, 0].view(n, n)
#     return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10, help='size')
    parser.add_argument('--epochs', type=int, default=1000, help='epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='lr')
    parser.add_argument('--lambd', type=float, default=3.88, help='lambd')
    args = parser.parse_args()
    n = args.n
    lr = args.lr
    epochs = args.epochs
    lambd = args.lambd
    device = 'cuda:0'
    instances = 100
    steps = 50
    bs = 128
    tau = 10

    def logistic_map(x, lambd=lambd):
        return lambd * x * (1 - x)

    def one_step_dynamics(current_state, A, s=0.2, lambd=3.88, eps=1e-3, device='cpu'):
        """Given x_t (n by 1) and coupling A, produce x_t+1"""
        diags = A.sum(1)
        # print(diags)
        if isinstance(current_state, np.ndarray):
            noise = np.random.randn(n, 1) * eps
        else:
            noise = torch.randn(n, 1) * eps
            noise = noise.to(device)
        next_state = (1 - s) * logistic_map(current_state, lambd=lambd) + s / diags * A @ logistic_map(current_state, lambd=lambd)
        next_state += noise
        return next_state

    def gumbel_sample(x, temp=1, hard=False):
        logp = x.view(-1, 2)
        out = gumbel_softmax(logp, temp, hard)
        out = out[:, 0].view(n, n)
        return out

    # data generation
    # G = nx.random_regular_graph(4, n, seed=2050)
    G = nx.random_regular_graph(4, n)
    A = nx.adjacency_matrix(G)
    A = A.todense() 
    A = np.array(A) # adjacency matrix A

    # For every initialization, we run 50 steps
    s = np.random.rand(n, 1)
    simulates = np.empty((instances*steps, n, 1))

    for i in range(instances):
        for j in range(steps):
            s = one_step_dynamics(s, A=A)
            simulates[i*steps+j, :, :] = s
            if (j+1) % steps == 0:
                s = np.random.rand(n, 1)

    # We use current step to predict the next step, [# data, N, 2]
    data = np.empty((1, n, 2))
    for i in range(simulates.shape[0]):
        if (i+1) % steps != 0:
            temp = np.empty((1, n, 2))
            temp[:, :, 0] = simulates[i, :, :].squeeze()
            temp[:, :, 1] = simulates[i+1, :, :].squeeze()
            data = np.concatenate((data, temp), axis=0)

    data = torch.from_numpy(data[1:, :, :]).to(torch.float32)
    data = data.to(device)
    data_loader = DataLoader(data, batch_size=bs, shuffle=True)
    print('Data size:', data.shape)

    x = torch.randn(n, n, device=device)
    # x = -1 * torch.rand(n, n, 2) # unnormalized logits
    x.requires_grad = True
    optimizer = torch.optim.Adam([x], lr=lr)
    loss_arr = []
    error_arr = []

    for epoch in range(epochs):
        # Training
        for idx, data in enumerate(data_loader): # data size [bs, n, 2]
            logits = x
            # logits = torch.empty(n, n, 2, device=device)
            # logits[:, :, 0] = torch.log(torch.sigmoid(x))
            # logits[:, :, 1] = torch.log((1-torch.sigmoid(x))+1e-20)
            A_p = gumbel_sample(logits, hard=False, temp=tau)
            s_t = torch.unsqueeze(data[:, :, 1], 2)
            s_p = one_step_dynamics(torch.unsqueeze(data[:, :, 0], 2), A_p, device=device)
            loss = torch.mean((s_t - s_p) ** 2)
            loss_arr.append(loss.data.cpu().numpy())
    #         print('Epoch: %i\t Batch Num: %i\t Loss: %.5f' % (epoch, idx, loss.data))
            loss.backward()
            optimizer.step()
            # tau *= 0.999
        # Test
        A_p = gumbel_sample(logits, hard=True, temp=1)
        # print(A_p)
        A_p = A_p.data.cpu().numpy()
        mask = np.ones((n, n))
        np.fill_diagonal(mask, np.zeros(n))
        error = np.abs(A_p * mask - A).sum()/(n**2-n)
        error_arr.append(error)
        print('Epoch: %i\t Error rate: %.5f' % (epoch, error))
    print('Accuracy: %.2f' % (1-error)*100)
    return error

if __name__ == '__main__':
    results_arr = []
    for _ in range(10):
        result = main()
        results_arr.append(result)
    


        