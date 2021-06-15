import numpy as np
from torch.utils.data import Dataset

a = np.load("anchors.npy")
n = np.load("negatives.npy")
p = np.load("positives.npy")

a_his = np.load("a_his.npy")
n_his = np.load("n_his.npy")
p_his = np.load("p_his.npy")


class ContrastiveData(Dataset):

    def __init__(self):
        self.anchors = a
        self.postives = p
        self.negatives = n

        self.a_his = a_his
        self.p_his = p_his
        self.n_his = n_his

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        return self.anchors[idx, :], self.postives[idx, :], self.negatives[idx, :], \
    self.a_his[idx, :], self.p_his[idx, :], self.n_his[idx, :]


if __name__=="__main__":
    dataset = ContrastiveData()
    print(len(dataset))
    a, p, n, a_his, p_his, n_his = dataset[3]
    print(a.shape)
    print(n.shape)
    print(a_his.shape)
    print(n_his.shape)