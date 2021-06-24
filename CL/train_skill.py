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
from predict import PP


dataset = CLData()
train_loader = DataLoader(dataset=dataset, batch_size=512, shuffle=True)

model = SupEncoder(126)
criterion = SupLoss()
predictor = PP()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

model.to(device)
criterion.to(device)


def train(nums):
    loss_his = []
    em_loss_his = []
    pp_loss_his = []
    global_step = 1
    for i in range(nums):
        for idx, (data, labels, s0_b, ss_b, aa_b, pp_b) in enumerate(train_loader):
            feats, em_loss, embedding = model(data, labels)
            loss, mean_log_prob = criterion(feats, labels)
            skills = torch.ones_like(feats)
            pp_hat = predictor(s0_b, ss_b, labels, embedding, skills)
            # print(pp_hat)
            pp_loss = torch.dist(pp_hat, pp_b)
            pp_loss = pp_loss.item()
            # print(pp_loss)
            loss_his.append(loss.item())
            em_loss_his.append(em_loss.item())
            # pp_loss_his.append(pp_loss)

            # loss_total = em_loss + loss + pp_loss
            loss_total = em_loss + loss
            optimizer.zero_grad()
            loss_total.backward()
            # loss.backward()
            optimizer.step()
            # print(global_step)
            if global_step % 100 == 0:
                # writer.add_scalar("loss", loss.item(), global_step)
                print("Global step is {}, the loss is {}".format(global_step, loss_total))
            global_step += 1
    return loss_his, em_loss_his, pp_loss_his, embedding

loss_his, em_loss_his, pp_loss_his, embedding = train(2000)
torch.save(model, 'model.pkl')
plt.plot(np.arange(len(loss_his)), loss_his)
# plt.figure()
# plt.plot(np.arange(len(em_loss_his)), em_loss_his)
plt.show()
