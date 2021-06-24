import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class PP(nn.Module):

    def __init__(self, s0_hidden_size=128, ss_hidden_size=256, output_size=200):
        super(PP, self).__init__()

        self.s0_hidden_size = s0_hidden_size
        self.ss_hidden_size = ss_hidden_size
        self.output_size = output_size

        self.s0_encoder = nn.Sequential(
            nn.Linear(4, self.s0_hidden_size),
            nn.ReLU(),
            nn.Linear(self.s0_hidden_size, self.s0_hidden_size),
            nn.ReLU(),
            # nn.Linear(self.hidden_size, 4)
        )
        self.s0_mu = nn.Linear(self.s0_hidden_size, 4)
        self.s0_log_sigma = nn.Linear(self.s0_hidden_size, 4)

        self.ss_encoder = nn.Sequential(
            nn.Linear(22, self.ss_hidden_size),
            nn.ReLU(),
            nn.Linear(self.ss_hidden_size, self.ss_hidden_size),
            nn.ReLU(),
            # nn.Linear(self.hidden_size, 4)
        )
        self.ss_mu = nn.Linear(self.ss_hidden_size, 12)
        self.ss_log_sigma = nn.Linear(self.ss_hidden_size, 12)

        self.predict = nn.Sequential(
            nn.Linear(4 + 12 + 24, 400),
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

    def forward(self, s0, ss, labels, embedding, skills):
        labels = labels.type(torch.LongTensor)
        labels = F.one_hot(labels)
        labels = labels.type(dtype=skills.dtype)
        skills = torch.matmul(labels, embedding)
        s0, _, _ = self.get_s0(s0)
        ss, _, _ = self.get_ss(ss)

        pp_hat = self.predict(torch.cat((s0, ss, skills), 1))
        return pp_hat

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
