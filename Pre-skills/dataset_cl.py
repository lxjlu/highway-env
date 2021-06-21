import numpy as np
from torch.utils.data import Dataset
import torch

a = np.load("anchors.npy")
n = np.load("negatives.npy")
p = np.load("positives.npy")

a_his = np.load("a_his.npy")
n_his = np.load("n_his.npy")
p_his = np.load("p_his.npy")


class ContrastiveData(Dataset):

    def __init__(self):
        self.anchors = torch.Tensor(a).unsqueeze(dim=1)
        self.postives = torch.Tensor(p).unsqueeze(dim=1)
        self.negatives = torch.Tensor(n)

        self.item = torch.cat((self.anchors, self.postives, self.negatives), dim=1)

    def __len__(self):
        return self.anchors.shape[0]

    def __getitem__(self, idx):
        return self.item[idx, :]

# if __name__=="__main__":
#     dataset = ContrastiveData()
#     print(len(dataset))
#     a, p, n, a_his, p_his, n_his = dataset[:10]
#     print(a.shape)
#     print(n.shape)
#     print(a_his.shape)
#     print(n_his.shape)
