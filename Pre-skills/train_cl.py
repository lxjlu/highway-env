import numpy as np

from dataset_cl import ContrastiveData
from contrastive_learning import Encoder
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

dataset = ContrastiveData()
train_loader = DataLoader(dataset=dataset, batch_size=256, shuffle=True)
model = Encoder(100, 400, 0.5)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
experiment_name = "cl"
writer = SummaryWriter(f"runs/{experiment_name}")

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

model.to(device)

def train(nums):
    loss_his = []
    global_step = 1
    for epoch in range(nums):
        print("epoch is {}".format(epoch+1))
        for idx, data in enumerate(train_loader):
            temp, sim, loss = model.forward(data)
            loss_his.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step == 100:
                writer.add_scalar("loss", loss.item(), global_step)
                # print("Train epoch is {}, the loss is {}".format(epoch + 1, loss))
            global_step += 1
    return loss_his


# 保存
torch.save(model, 'model.pkl')
# 加载
# model = torch.load('\model.pkl')

loss_his = train(5000)
plt.plot(np.arange(len(loss_his)), loss_his)
plt.show()
