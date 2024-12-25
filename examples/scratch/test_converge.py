import torch

beta = 0.1
N = 10
M = 5
rewards = torch.randint(0, 2, (N,))
l_ref = torch.zeros((N,))
l = torch.zeros_like(l_ref, requires_grad=True)
