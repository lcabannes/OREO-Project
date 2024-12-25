import torch
import numpy as np
import matplotlib.pyplot as plt


def plot(name: str, idx: int = 0):
    a = torch.load(f"weights_{name}.pt")
    plt.plot(sorted(a[idx]), np.linspace(0, 1, len(a[idx]), endpoint=False), label=name)


names = ["kl", "fkl", "is", "is-kl", "is-fkl", "actor-lr5e-7", "actor-lr5e-7-is", "actor-lr5e-7-kl"]
for name in names:
    plot(name, 1)
# plt.xlim(-5, 100)
plt.xlim(-0.1, 10)
plt.legend()
plt.savefig("weights-neg.jpg")
