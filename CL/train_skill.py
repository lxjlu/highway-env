from cl_features import SupEncoder
from cl_loss import SupLoss
from cl_data import CLData

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


dataset = CLData()
train_loader = DataLoader(dataset=dataset, batch_size=512, shuffle=True)

model = SupEncoder(126)
criterion = SupLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

model.to(device)
criterion.to(device)


def train(nums):
    loss_his = []
    global_step = 1
    for i in range(nums):
        for idx, (data, labels) in enumerate(train_loader):
            feats = model(data)
            loss, mean_log_prob = criterion(feats, labels)
            loss_his.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(global_step)
            if global_step % 100 == 0:
                # writer.add_scalar("loss", loss.item(), global_step)
                print("Global step is {}, the loss is {}".format(global_step, loss))
            global_step += 1
    return loss_his

loss_his = train(2000)
torch.save(model, 'model.pkl')
plt.plot(np.arange(len(loss_his)), loss_his)
plt.show()
