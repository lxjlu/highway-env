from nets import ReplayBuffer
import numpy as np
import torch

# 加载
clpp = torch.load("clpp_model.pkl")
clpp.eval = True
clde = torch.load("clde_model.pkl")
em = np.load("embedding.npy")
em = torch.Tensor(em)
bf = ReplayBuffer(load_pid=True)
z_his, u_his, s0_his, X_his, ss_his, cl_his, road_r_his = bf.sample(1)

s0_his = torch.Tensor(s0_his)
ss_his = torch.Tensor(ss_his)
labels_his = torch.Tensor(z_his)
pp_hat, s0_mu, ss_mu, s0_log_sigma, ss_log_sigma = clpp(s0_his, ss_his, labels_his, em)

pp = pp_hat.detach().cpu().numpy()

