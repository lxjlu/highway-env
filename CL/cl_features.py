import torch
from torch import nn
import torch.nn.functional as F


class SupEncoder(nn.Module):
    """
    把初始状态+路面曲线+动作作为输入，输出为隐藏空间表示
    """

    def __init__(self, input_dim, hidden_dim=512, head='mlp', feat_dim=24):
        super(SupEncoder, self).__init__()
        self.feat_dim = feat_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.embedding_dim = feat_dim
        self.num_embedding = 9
        self._embedding = nn.Embedding(self.num_embedding, self.embedding_dim)
        self._embedding.weight.data.uniform_(-1/self.num_embedding, 1/self.num_embedding)



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

    def forward(self, x, label):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        loss_em = self.emb_loss(feat, label)
        return feat, loss_em, self._embedding.weight

    def emb_loss(self, feats, labels):
        labels = labels.contiguous().view(-1, 1)
        loss = 0
        for i in range(9):
            mask = torch.eq(labels, i)
            tt = feats.masked_select(mask).view(-1, self.feat_dim)

            # print(feats.shape)
            loss += torch.dist(tt.detach(), self._embedding.weight[i, :], 2)
        return loss
