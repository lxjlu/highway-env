import numpy as np
from sklearn.manifold import TSNE
import torch
import matplotlib.pyplot as plt

data = np.load("data.npy")
# data = data[:1000]
label = np.load("label.npy")
# label = label[:1000]

data = torch.Tensor(data)
label = torch.Tensor(label)
model = torch.load('model.pkl')
X, loss_em, embedding_weights = model(data, label)
X = X.detach().numpy()
X_embedded = TSNE(n_components=2).fit_transform(X)

embedding = embedding_weights.detach().numpy()
np.save("embedding.npy", embedding)
Y_embedded = TSNE(n_components=2).fit_transform(embedding)

# plt.scatter(X[:, 0], X[:, 1])
plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
# plt.figure()
# plt.scatter(Y_embedded[:, 0], Y_embedded[:, 1])
plt.show()


