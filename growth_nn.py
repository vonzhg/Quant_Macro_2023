# solve optimal growth model using neural network.
# this version: 03.02.2022

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.optimize import minimize_scalar
from datetime import datetime

start_time = datetime.now()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

n_episode = 500
n_batch = 50
n_epoch = 1500
lr_rate = 0.001
alpha = 0.33
beta = 1.0
delta = 0.9
c_eps = 1e-5
k_size = n_episode
k_min = 1e-4 + 0.1
k_max = 2.5
k0_in = 0
v_scale = -50
k_vec = torch.linspace(k_min, k_max, k_size)
k_vec = k_vec.view(k_size, 1)
k_vec = k_vec.to(device)

# construction of neural network
n1 = 10
n2 = 50
decision_rule = nn.Sequential(nn.Linear(1, n1),
                              nn.ReLU(),
                              nn.Linear(n1, n2),
                              nn.ReLU(),
                              nn.Linear(n2, 1))
# nn.Sigmoid())
decision_rule = decision_rule.to(device)


def dv1_fun(x_k):
    h = 1e-9
    h = torch.FloatTensor([h])
    h = h.to(device)
    x_k = torch.FloatTensor([x_k])
    x_k = x_k.to(device)

    x_k1 = x_k + h
    x_k1 = x_k1.to(device)
    x_gg1 = decision_rule(x_k1)
    x_gg1 = x_gg1.to(device)
    x_v1 = x_gg1[1] * v_scale
    x_v1 = x_v1.to(device)

    x_k2 = x_k - h
    x_k2 = x_k2.to(device)
    x_gg2 = decision_rule(x_k2)
    x_gg2 = x_gg2.to(device)
    x_v2 = x_gg2[1] * v_scale
    x_v2 = x_v2.to(device)

    return (x_v1 - x_v2) / (2 * h)


def sim_k(x_k0):

    #################################
    # random simulation
    x_k0 = (k_max - k_min) * torch.rand(n_episode, 1) + k_min
    x_gg0 = decision_rule(x_k0)
    x_k1 = x_gg0
    x_c0_orig = x_k0 ** alpha - x_k1
    x_c0 = torch.maximum(x_c0_orig, c_eps_tensor)

    x_gg1 = decision_rule(x_k1)
    x_k2 = x_gg1
    x_c1_orig = x_k1 ** alpha - x_k2
    x_c1 = torch.maximum(x_c1_orig, c_eps_tensor)
    x_r1 = alpha * x_k1 ** (alpha - 1)

    x_f_ee = (x_c1 / (x_c0 * x_r1 * beta * delta) - 1)
    return x_k1, x_f_ee


loss_function = torch.nn.MSELoss(reduction='mean')

# ++++++++++++++++++++++++++++++++++++++++++++++++
# Neural network
# ++++++++++++++++++++++++++++++++++++++++++++++++

f_ee_sim = torch.zeros((k_size, 1), device=device)
f_ee_true = torch.zeros_like(f_ee_sim, device=device)

optimizer = torch.optim.Adam(decision_rule.parameters(), lr=lr_rate)
c_eps_tensor = torch.ones_like(k_vec) * c_eps
zero_tensor = torch.zeros_like(k_vec)
losses = []

for i in range(n_epoch):
    sim_k0 = (k_max - k_min) * torch.rand(n_episode, 1) + k_min
    sim_k1, f_ee_sim = sim_k(sim_k0)
    sim_k1 = sim_k1.to(device)

    loss = loss_function(f_ee_true, f_ee_sim)
    print(loss.item())
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

decision_rule.state_dict()

# ++++++++++++++++++++++++++++++++++++++++++++++++
# Plot value functions


gg_nn = decision_rule(k_vec)
gg_nn = gg_nn.to(device)
gk_nn = gg_nn[:, 0].view(k_size, 1)
# gv_nn = gg_nn[:,1].view(k_size, 1) * v_scale
gk_nn = gk_nn.to(device)
# gv_nn = gv_nn.to(device)
gk_nn = gk_nn.cpu().detach().numpy()
# gv_nn = gv_nn.cpu().detach().numpy()

tmp1 = np.log((1 - delta * alpha) / (1 - delta * alpha + beta * delta * alpha))
tmp2 = np.log(beta * delta * alpha / (1 - delta * alpha + beta * delta * alpha))
tmp3 = delta * alpha / (1 - delta * alpha) * tmp2
k_vec = k_vec.cpu().detach().numpy()
gv_close = 1 / (1 - delta) * (tmp1 + tmp3) + alpha / (1 - delta * alpha) * np.log(k_vec)
# gv_close = gv_close.to(device)

plt.figure(figsize=(11, 8))

plt.subplot(211)
plt.plot(k_vec, gk_nn, k_vec, alpha * delta * k_vec ** alpha, lw=2, alpha=0.6, label='policy')
plt.xlabel('capital')
plt.ylabel('policy')
plt.legend(loc='upper right')

plt.subplot(212)
plt.plot(np.log(losses))
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()

plt.savefig('optgrowth.jpg')

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))