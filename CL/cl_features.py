import torch
from torch import nn
import torch.nn.functional as F


class SupEncoder(nn.Module):
    """
    把初始状态+路面曲线+动作作为输入，输出为隐藏空间表示
    """

    def __init__(self, input_dim, hidden_dim=512, head='mlp', feat_dim=24):
        super(SupEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.embedding_dim = feat_dim
        self.num_embedding = 9
        self._embedding = nn.Embedding(self.num_embedding, self.embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)



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
