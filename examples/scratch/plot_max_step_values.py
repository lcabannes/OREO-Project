import matplotlib.pyplot as plt
import torch

a = torch.load("max_step_value.pt")
a = sorted(a)
plt.plot(a, list(range(len(a))))
a = torch.load("max_step_value_freeze_2.pt")
a = sorted(a)
plt.plot(a, list(range(len(a))), label="freeze")
a = torch.load("max_step_value_token.pt")
a = sorted(a)
plt.plot(a, list(range(len(a))), label="token")
plt.legend()
plt.savefig("max_step_value.jpg")
