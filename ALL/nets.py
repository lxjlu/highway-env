import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import collections
import random
import os


class ReplayBuffer():
    def __init__(self, buffer_limit=50000):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.ntr = 0

    def put(self, one_data):
        # (z, u, s0, X, ss)
        # z: 0, 1, 2, 3 .., 8
        # u: omega_his + accel_his
        # s0: x, y, theta, v
        # X: x_his + y_his + theta_his + v_his  1 - N+1
        # ss: road_x + road_y
        self.buffer.append(one_data)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        z_his, u_his, s0_his, X_his, ss_his = [], [], [], [], []
        cl_his = []

        for one_data in mini_batch:
            z, u, s0, X, ss = one_data
            z_his.append(z)
            u_his.append(u)
            s0_his.append(s0)
            X_his.append(X)
            ss_his.append(ss)

            cl_input = s0 + ss + u
            cl_his.append(cl_input)

        return np.array(z_his), np.array(u_his), np.array(s0_his), np.array(X_his), \
               np.array(ss_his), np.array(cl_his)


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
