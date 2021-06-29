import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import collections
import random
import os
from torch.autograd import Variable


class ReplayBuffer():
    def __init__(self, load_pid=False, buffer_limit=50000):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.ntr = 0

        if load_pid:
            label_his = np.load("label_his.npy")
            aa_his = np.load("aa_his_n.npy")
            s0_his = np.load("s0_his_n.npy")
            pp_his = np.load("pp_his_n.npy")
            ss_his = np.load("ss_his_n.npy")
            road_r_his = np.load("road_r_his.npy")

            for i in range(label_his.shape[0]):
                one_data = (label_his[i], aa_his[i], s0_his[i], pp_his[i], ss_his[i], road_r_his[i])
                self.put(one_data)


    def put(self, one_data):
        # (z, u, s0, X, ss)
        # z: 0, 1, 2, 3 .., 8
        # u: omega_his + accel_his
        # s0: x, y, theta, v
        # X: x_his + y_his + theta_his + v_his  1 - N+1
        # ss: road_x + road_y
        self.buffer.append(one_data)
        self.ntr += 1

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        z_his, u_his, s0_his, X_his, ss_his = [], [], [], [], []
        cl_his = []
        road_r_his = []

        for one_data in mini_batch:
            z, u, s0, X, ss, road_r = one_data
            z_his.append(z)
            u_his.append(u)
            s0_his.append(s0)
            X_his.append(X)
            ss_his.append(ss)
            road_r_his.append(road_r)

            cl_input = np.concatenate((s0, ss, u), 0)
            cl_his.append(cl_input)

        return np.array(z_his), np.array(u_his), np.array(s0_his), np.array(X_his), \
               np.array(ss_his), np.array(cl_his), np.array(road_r_his)


class CLEncoder(nn.Module):
    """
    s0+ss+u -> zz
    """
    def __init__(self, input_dim=126, hidden_dim=512, head='mlp', feat_dim=24):
        super(CLEncoder, self).__init__()

        self.feat_dim = feat_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        if head == 'linear':
            self.head = nn.Linear(hidden_dim, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

class CLLoss(nn.Module):

    def __init__(self, temperature=0.5):
        super(CLLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        featuers: hidden vector of shap [bsz, 128]
        labels: ground truth of shap [bsz]
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')

        # 这里mask是什么, [bsz, bsz] 对角为True其他为False
        mask_label = torch.eq(labels, labels.T).float().to(device)
        # mask_pos = mask_label.masked_select(~torch.eye(batch_size).bool()).view(batch_size, -1)

        sim = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        sim_label = sim * mask_label
        sim_pos = sim_label * (~torch.eye(batch_size).bool())
        sim_neg = sim * (~mask_label.bool())

        # log
        exp_sim_pos = torch.exp(sim_pos)
        exp_sim_neg = torch.exp(sim_neg)

        log_prob = torch.log(exp_sim_pos.sum(1, keepdim=True) / exp_sim_neg.sum(1, keepdim=True))
        mean_log_prob = log_prob / (mask_label.sum(1, keepdim=True) - 1)

        # loss
        loss = -mean_log_prob
        loss = loss.view(batch_size, -1).mean()

        return loss

class CLPP(nn.Module):

    def __init__(self, eval=False):
        super(CLPP, self).__init__()
        self.eval = eval
        self.s0_encoder = nn.Sequential(
            nn.Linear(4, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU()
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
            nn.Linear(400, 200),
            nn.Tanh()
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

    def forward(self, s0, ss, labels, em):
        labels = labels.type(torch.LongTensor)
        labels = F.one_hot(labels, 9) # 会不会有影响？？
        labels = labels.type(torch.float)
        skills = torch.matmul(labels, em.detach())
        s0, s0_mu, s0_log_sigma = self.get_s0(s0)
        ss, ss_mu, ss_log_sigma = self.get_ss(ss)

        if self.eval:
            pp_hat = self.predict(torch.cat((s0_mu, ss_mu, skills), 1))
        else:
            pp_hat = self.predict(torch.cat((s0, ss, skills), 1))
        return pp_hat, s0_mu, ss_mu, s0_log_sigma, ss_log_sigma

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        # std = logvar.mul(0.5).exp_()
        # eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

class CLDeconder(nn.Module):

    def __init__(self):
        super(CLDeconder, self).__init__()

        self.deconder = nn.Sequential(
            nn.Linear(4+22+24, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 100),
            nn.Tanh()
        )

    def forward(self, s0, ss, hidden):
        actions = self.deconder(torch.cat((s0, ss, hidden), 1))
        return actions


class ActionNet(nn.Module):
    def __init__(self, num_goal_states=2, num_around_states=4):
        super(ActionNet, self).__init__()

        self.action_prob = nn.Sequential(
            nn.Linear(num_goal_states+num_around_states, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 9),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        prob = self.action_prob(torch.cat((g, s), 1))
        z = torch.argmax(prob)
        return z, prob

