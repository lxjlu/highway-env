from dataset_cl import ContrastiveData
from contrastive_learning import Encoder
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

dataset = ContrastiveData()
train_loader = DataLoader(dataset=dataset, batch_size=256, shuffle=True)
model = Encoder(100, 400, 0.5)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(nums):
    for epoch in range(nums):

        for idx, data in enumerate(train_loader):
            temp, sim, loss = model.forward(data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx == 10:
                print("Train epoch is {}, the loss is {}".format(epoch + 1, loss))





train(100)
