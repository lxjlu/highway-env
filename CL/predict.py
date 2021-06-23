import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class PP(nn.Module):

    def __init__(self, embedding, hidden_size, output_size):
        super(PP, self).__init__()

        self.embedding = embedding
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, s0, ss, z):
        skill = self.embedding[z]

