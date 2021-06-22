import numpy as np
from sklearn.manifold import TSNE
import torch
import matplotlib.pyplot as plt

data = np.load("data.npy")
# data = data[:1000]
label = np.load("label.npy")
# label = label[:1000]

data = torch.Tensor(data)
model = torch.load('model.pkl')
X = model(data)
X = X.detach().numpy()
X_embedded = TSNE(n_components=2).fit_transform(X)

plt.scatter(X[:, 0], X[:, 1])
plt.show()


