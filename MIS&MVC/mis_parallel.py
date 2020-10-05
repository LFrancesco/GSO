import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch.nn.functional import gumbel_softmax
from utils import gumbel_softmax_3d


def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    G = nx.from_dict_of_lists(data)
    return G


def train(args, G):
    bs = args.batch_size
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G)
    A = torch.from_numpy(A.todense())
    A = A.type(torch.float32)

    # set parameters
    if torch.cuda.is_available():
        A = A.cuda()
        x = torch.randn(bs, n, 1, device='cuda')*1e-5
        x.requires_grad = True
    else:
        x = torch.randn(bs, n, 1, requires_grad=True)*1e-5

    # set optimizer
    optimizer = torch.optim.SGD([x], lr=args.lr)

    # training
    cost_arr = []
    for _ in range(args.iterations):
        optimizer.zero_grad()
        if torch.cuda.is_available():
            probs = torch.empty(bs, n, 2, device='cuda')
        else:
            probs = torch.empty(bs, n, 2)
        p = torch.sigmoid(x)
        probs[:, :, 0] = p.squeeze()
        probs[:, :, -1] = 1-probs[:, :, 0]
        logits = torch.log(probs+1e-10)
        s = gumbel_softmax_3d(logits, tau=args.tau, hard=args.hard)[:, :, 0]
        s = torch.unsqueeze(s, -1)  # size [bs, n, 1]
        cost = -1 * torch.sum(s)
        constraint = torch.sum(torch.transpose(s, 1, 2) @ A @ s)
        loss = cost + args.eta * constraint
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            constraint = torch.squeeze(torch.transpose(s, 1, 2) @ A @ s)
            constraint = constraint.cpu().numpy()
            idx = np.argwhere(constraint == 0)  # select constraint=0
            if len(idx) != 0:
                s = gumbel_softmax_3d(logits, tau=args.tau, hard=True)[:, :, 0]
                s = torch.unsqueeze(s, -1).cpu()  # size [bs, n, 1]
                cost = -1 * torch.sum(s, dim=1)[idx.reshape(-1,)]
                # from size [bs, 1] select constrain=0
                cost_arr.append(torch.min(cost.cpu()))

    return cost_arr


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Solving MIS problems (with fixed tau in GS, parallel version)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size (default: 128)')
    parser.add_argument('--data', type=str, default='cora',
                        help='data name (default: cora)')
    parser.add_argument('--tau', type=float, default=1.,
                        help='tau value in Gumbel-softmax (default: 1)')
    parser.add_argument('--hard', type=bool, default=True,
                        help='hard sampling in Gumbel-softmax (default: True)')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate (default: 1e-2)')
    parser.add_argument('--eta', type=float, default=5.,
                        help='constraint (default: 5)')
    parser.add_argument('--ensemble', type=int, default=100,
                        help='# experiments (default: 100)')
    parser.add_argument('--iterations', type=int, default=20000,
                        help='# iterations in gradient descent (default: 20000)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    args = parser.parse_args()

    # torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    # loading data
    G = load_data('./data/ind.' + args.data + '.graph')

    for i in range(args.ensemble):
        cost = train(args, G)
        if len(cost) != 0:
            print('# {}, cost: {}'.format(i, min(cost)))
        else:
            print('Failed!')


if __name__ == '__main__':
    main()
