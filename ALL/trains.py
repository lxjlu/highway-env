from nets import CLEncoder, CLLoss, ReplayBuffer, CLEmbedding

import torch
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

rb = ReplayBuffer()
clen = CLEncoder()
clloss = CLLoss()
clembeeding = CLEmbedding()

CL_optimizer = optim.Adam(clen.parameters(), lr=1e-3)
EM_optimizer = optim.Adam(clen.parameters(), lr=1e-3)


def train_CL(nums=100, batch_size=512):
    loss_his = []
    global_step = 1
    for i in range(nums):
        mini_s0, mini_ss, mini_A, mini_X, mini_l, mini_d = rb.sample(batch_size)
        data = torch.Tensor(mini_d)
        label = torch.Tensor(mini_l)
        feat = clen(data)
        loss = clloss(feat, label)
        loss_his.append(loss.item())

        CL_optimizer.zero_grad()
        loss.backward()
        CL_optimizer.step()

        if global_step % 100 == 0:
            print("global step is {}, avg loss is {}".format(global_step, np.mean(loss_his[-100:])))
        global_step += 1


    return loss_his

def train_embedding(nums=100, batch_size=512):
    loss_his = []
    global_step = 1
    for i in range(nums):
        mini_s0, mini_ss, mini_A, mini_X, mini_l, mini_d = rb.sample(batch_size)
        data = torch.Tensor(mini_d)
        label = torch.Tensor(mini_l)
        feat = clen(data)
        loss = clembeeding(feat, label)
        loss_his.append(loss.item())

        EM_optimizer.zero_grad()
        loss.backward()
        EM_optimizer.step()

        if global_step % 100 == 0:
            print("global step is {}, avg loss is {}".format(global_step, np.mean(loss_his[-100:])))
        global_step += 1

    return loss_his, clembeeding._embedding.weight

def train_pp(nums=100):
    loss_his = []
    for i in range(nums):
        pass


train_CL(nums=1)