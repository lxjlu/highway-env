import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from nets import ReplayBuffer

"""
用来排除AVE loss中的nan问题：原因是exp()太大，所以要在logvar输出中加入
初始化权重，使得不那么大
"""
load_pid = True
bf = ReplayBuffer(load_pid)
em = np.load("embedding.npy")


class PP(nn.Module):

    def __init__(self):
        super(PP, self).__init__()

        self.s0_encoder = nn.Sequential(
            nn.Linear(4, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
        )
        self.s0_mu = nn.Linear(400, 2)
        self.s0_log_sigma = nn.Linear(400, 2)
        torch.nn.init.uniform_(self.s0_log_sigma.weight, a=-0.0001, b=0.0001)

        self.ss_encoder = nn.Sequential(
            nn.Linear(22, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU()
        )
        self.ss_mu = nn.Linear(400, 8)
        self.ss_log_sigma = nn.Linear(400, 8)
        torch.nn.init.uniform_(self.ss_log_sigma.weight, a=-0.0001, b=0.0001)

        self.predict = nn.Sequential(
            nn.Linear(2 + 8 + 24, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 200)
        )

    def get_s0(self, s0):
        s0 = self.s0_encoder(s0)
        s0_mu, s0_log_sigma = self.s0_mu(s0), self.s0_log_sigma(s0)
        z = self.reparameterize(s0_mu, s0_log_sigma)
        return z, s0_mu, s0_log_sigma

    def get_ss(self, ss):
        ss = self.ss_encoder(ss)
        ss_mu, ss_log_sigma = self.ss_mu(ss), self.ss_log_sigma(ss)
        z = self.reparameterize(ss_mu, ss_log_sigma)
        return z, ss_mu, ss_log_sigma

    def forward(self, s0, ss, labels):
        labels = labels.type(torch.LongTensor)
        labels = F.one_hot(labels)
        labels = labels.type(torch.float)
        skills = torch.matmul(labels, torch.Tensor(em))

        s0, s0_mu, s0_log_sigma = self.get_s0(s0)
        ss, ss_mu, ss_log_sigma = self.get_ss(ss)

        pp_hat = self.predict(torch.cat((s0, ss, skills), 1))
        return pp_hat, s0_mu, ss_mu, s0_log_sigma, ss_log_sigma


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        # std = logvar.mul(0.5).exp_()
        # eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)




clpp = PP()
pp_optimizer = optim.Adam(clpp.parameters(), lr=1e-5)

for i in range(5001):
    z_his, u_his, s0_his, X_his, ss_his, cl_his = bf.sample(512)
    s0_his = torch.Tensor(s0_his)
    ss_his = torch.Tensor(ss_his)
    labels_his = torch.Tensor(z_his)
    X_his = torch.Tensor(X_his)
    pp_hat, s0_mu, ss_mu, s0_log_sigma, ss_log_sigma = clpp(s0_his, ss_his, labels_his)
    X_F = X_his.reshape(512, -1, 50)
    X_F = X_F[:, :, -1]
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_s0 = -0.5 * torch.sum(1 + s0_log_sigma - s0_mu.pow(2) - s0_log_sigma.exp())
    KLD_ss = -0.5 * torch.sum(1 + ss_log_sigma - ss_mu.pow(2) - ss_log_sigma.exp())
    # loss = 0.0001 * (torch.dist(pp_hat, X_his, p=0) + KLD_s0 + KLD_ss)
    loss = torch.dist(pp_hat, X_his) + 0.001 * (KLD_s0 + KLD_ss)
    pp_optimizer.zero_grad()
    loss.backward()
    pp_optimizer.step()

    if i % 100 == 0:
        print("第 {} 次的 loss 是 {}".format(i + 1, loss.item()))
