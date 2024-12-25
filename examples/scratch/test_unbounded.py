import torch
from torch import optim
from torch.functional import F


def loss_fn(v: torch.Tensor, l: torch.Tensor, l_ref: torch.Tensor, beta: float):
    logps = l.log_softmax(dim=0)
    reference_logps = l_ref.log_softmax(dim=0)
    loss1 = torch.square(v - (-1 - beta * (logps[0] - reference_logps[0])))
    loss2 = torch.square(v - (1 - beta * (logps[1] - reference_logps[1])))
    # loss1 = loss1 * (logps[0] - reference_logps[0]).detach().clip(max=1).exp()
    # loss2 = loss2 * (logps[1] - reference_logps[1]).detach().clip(max=1).exp()
    loss = (loss1 + loss2) / 2
    # kls = torch.exp(reference_logps - logps) + (logps - reference_logps) - 1
    # kls = torch.square(logps - reference_logps)
    # kl = (kls[0] + kls[1]) / 2
    # kl = F.kl_div(reference_logps, logps, reduction="none", log_target=True).sum()
    # kl = F.kl_div(logps, reference_logps, reduction="none", log_target=True).sum()
    # loss = loss + 0.01 * kl
    return (
        loss,
        (logps[0] - reference_logps[0]).detach().exp(),
        (logps[1] - reference_logps[1]).detach().exp(),
    )


beta = 0.1
v = torch.tensor(0.0, requires_grad=True)
l_ref = torch.zeros((3,))
l = torch.zeros_like(l_ref, requires_grad=True)
# l_ref = torch.tensor(range(99, -1, -1), dtype=torch.float)
# l_ref[1] = 10
# l = l_ref.clone().requires_grad_(True)

lr = 0.001
v_optim = optim.Adam([v], lr)
l_optim = optim.Adam([l], lr)
# lr = 0.001
# v_optim = optim.Adam([v], lr, betas=(0.9, 0.5))
# l_optim = optim.Adam([l], lr, betas=(0.9, 0.5))
# lr = 0.1
# v_optim = optim.SGD([v], lr, momentum=0.9)
# l_optim = optim.SGD([l], lr, momentum=0.9)

cnt = 0
while True:
    if v.item() > 0.8:
        __import__("pdb").set_trace()

    loss, _, _ = loss_fn(v, l.detach(), l_ref, beta)
    v_optim.zero_grad()
    loss.backward()
    v_optim.step()

    loss, p, q = loss_fn(v.detach(), l, l_ref, beta)
    l_optim.zero_grad()
    loss.backward()
    l_optim.step()

    cnt += 1
    if cnt % 100 == 0:
        print("{:.10f} {:.10f} {:.10f} {:.10f}".format(v.item(), p.item(), q.item(), loss.item()))
