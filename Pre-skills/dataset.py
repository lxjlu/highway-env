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
        self.anchors = torch.Tensor(a)
        self.postives = torch.Tensor(p)
        self.negatives = torch.Tensor(n)

        self.a_his = torch.Tensor(a_his)
        self.p_his = torch.Tensor(p_his)
        self.n_his = torch.Tensor(n_his)

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        return self.anchors[idx, :], self.postives[idx, :], self.negatives[idx, :], \
    self.a_his[idx, :], self.p_his[idx, :], self.n_his[idx, :]


# if __name__=="__main__":
#     dataset = ContrastiveData()
#     print(len(dataset))
#     a, p, n, a_his, p_his, n_his = dataset[:10]
#     print(a.shape)
#     print(n.shape)
#     print(a_his.shape)
#     print(n_his.shape)