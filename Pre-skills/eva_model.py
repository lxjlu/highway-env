import torch
import numpy as np
from dataset_cl import ContrastiveData
from torch.utils.data import DataLoader

dataset = ContrastiveData()
eval_loader = DataLoader(dataset=dataset, batch_size=256, shuffle=True)

model = torch.load('model.pkl')
for idx, data in enumerate(eval_loader):
    temp, sim, loss = model.forward(data)
    print(sim)