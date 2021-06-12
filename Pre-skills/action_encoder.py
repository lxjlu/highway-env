import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionEncoder(nn.Module):

    def __init__(self, horizon, state_embed_size, road_state_num, road_embed_size, skill_embed_size=9,
                 state_num=4, action_num=2, hidden_size=400):
        super(ActionEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.actions_num = horizon * action_num
        self.state_num = state_num
        self.state_embed_size = state_embed_size
        self.road_state_num = road_state_num
        self.road_embed_size = road_embed_size
        self.skill_embed_size = skill_embed_size

        self.s0_encoder = nn.Sequential(
            nn.Linear(self.state_num, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            # nn.Linear(self.hidden_size, self.state_embed_size)
        )
        self.s0_mu = nn.Linear(self.hidden_size, self.state_embed_size)
        self.s0_log_sigma = nn.Linear(self.hidden_size, self.state_embed_size)

        self.ss_encoder = nn.Sequential(
            nn.Linear(self.road_state_num, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            # nn.Linear(self.hidden_size, self.road_embed_size)
        )
        self.ss_mu = nn.Linear(self.hidden_size, self.road_embed_size)
        self.ss_log_sigma = nn.Linear(self.hidden_size, self.road_embed_size)

        self.skill_encoder = nn.Sequential(
            nn.Linear(self.actions_num + self.road_state_num + self.state_num, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.skill_embed_size),
            nn.Softmax()
        )

    def encode(self):
        pass

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def predictor(self, s_0, s_s, z):
        pass

    def forward(self, s_0, s_s, A):
        skill_embed = self.skill_embed_size(s_0 + s_s + A)
        return skill_embed


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
