import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset_cl import ContrastiveData


class Encoder(nn.Module):

    def __init__(self, embed_size, hidden_size, temperature):
        super(Encoder, self).__init__()

        self.skill_embed_size = embed_size
        self.hidden_size = hidden_size
        self.t = temperature

        self.skill_encoder = nn.Sequential(
            nn.Linear(126, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.skill_embed_size),
            # nn.Softmax()
        )

    def forward(self, x):
        b, n_item, size = x.shape
        x = x.reshape(-1, size)
        temp = self.skill_encoder(x)
        temp = temp.reshape(b, n_item, self.skill_embed_size)
        sim = F.cosine_similarity(temp[:, 0, :].unsqueeze(1), temp, dim=-1)
        sim = torch.exp(sim[:, 1:])
        loss = sim[:, 0] / torch.sum(sim, dim=1)
        return temp, sim, -loss.mean()


dataset = ContrastiveData()
tt = dataset[:2]
en = Encoder(100, 400, 0.5)
temp, sim, loss = en.forward(tt)
