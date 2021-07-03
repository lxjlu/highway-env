import gym
import highway_env
import numpy as np
from highway_env.utils import lmap
import torch
from nets import ReplayBuffer, CLEncoder, CLLoss, CLPP, CLDeconder
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F


bf = ReplayBuffer(load_pid=True)
# clen = CLEncoder()
# clloss = CLLoss()
clpp = CLPP()
clde = CLDeconder()
clen = torch.load("clen_model.pkl")
clloss = torch.load("clloss_model.pkl")
# clpp = torch.load("clpp_model.pkl")
# clde = torch.load("clde_model.pkl")

# embedding = torch.nn.Embedding(9, 24)
cl_optimizer = optim.Adam(clen.parameters(), lr=1e-3)
# em_optimizer = optim.Adam(embedding.parameters(), lr=1e-3)
pp_optimizer = optim.Adam(clpp.parameters(), lr=1e-3)
de_optimizer = optim.Adam(clde.parameters(), lr=1e-3)
em = np.load("embedding.npy")


def pid_collector(nums=10000, save_pid_data=False, bf=None):
    s0_his, ss_his, aa_his, tt_his, pp_his, label_his = [], [], [], [], [], []
    s0_his_n, ss_his_n, aa_his_n, tt_his_n, pp_his_n = [], [], [], [], []
    road_r_his = []
    max_v = 30
    max_omega = np.pi/3
    max_a = 5  # [m/s]
    for i in range(nums):
        print("pid collector 第 {} 次收集".format(i + 1))
        env = gym.make("myenv-r1-v0")
        labels_index = np.arange(9).reshape(3, 3)
        N = 50

        lanes_count = env.config["lanes_count"]
        lane_id = np.random.choice(np.arange(lanes_count))

        if lane_id == 0:
            target_lane_id = np.random.choice([0, 1])
        elif lane_id == lanes_count - 1:
            target_lane_id = np.random.choice([lanes_count - 1, lanes_count - 2])
        else:
            target_lane_id = np.random.choice([lane_id - 1, lane_id, lane_id + 1])

        lon_operation = np.random.choice([0, 1, 2])  # 1保持 0减速 2加速
        v_lane_id = ("a", "b", lane_id)
        target_lane_id2 = ("a", "b", target_lane_id)
        v_target_s = (lon_operation - 1) * 5 + env.vehicle.speed
        v_target_s = np.clip(0, 30, v_target_s)

        positon_x = np.random.choice(np.arange(0, env.road.network.get_lane(v_lane_id).length, 5))
        positon_y = np.random.choice(np.arange(-2, 2.1, 0.5))
        heading = np.random.choice(
            env.road.network.get_lane(v_lane_id).heading_at(positon_x) + np.arange(-np.pi / 12, np.pi / 12, 10))
        speed = np.random.choice(np.arange(0, 25, 2))

        env.config["v_lane_id"] = v_lane_id
        env.config["v_target_id"] = target_lane_id2
        env.config["v_x"] = positon_x
        env.config["v_y"] = positon_y
        env.config["v_h"] = heading
        env.config["v_s"] = speed
        env.config["v_target_s"] = v_target_s

        env.reset()

        p = env.vehicle.position
        i_h = env.vehicle.heading
        i_s = env.vehicle.speed
        temp = [p[0], p[1], i_h, i_s]
        # radius*(end_phase - start_phase) * self.direction
        road_length = env.road.network.get_lane(v_lane_id).length
        road_r = env.config["radius"]
        road_r_his.append(road_r)
        temp_n = [lmap(p[0], [-(road_r+4), (road_r+4)], [-1, 1]), lmap(p[1], [0, 2*(road_r+4)], [-1, 1]),
                lmap(i_h, [-2*np.pi, 2*np.pi], [-1, 1]), lmap(i_s, [-max_v, max_v], [-1, 1])]
        x_road, y_road, x_road_n, y_road_n = env.vehicle.target_lane_position(p, road_r)

        action_his_omega = []
        action_his_omega_n = []
        action_his_accel = []
        action_his_accel_n = []
        action = 1
        x_his, y_his, h_his, s_his = [], [], [], []
        x_his_n, y_his_n, h_his_n, s_his_n = [], [], [], []

        for _ in range(N):
            # 应该有一个判断是不是出去了的函数

            env.step(action)
            action_his_omega.append(env.vehicle.action["steering"])
            action_his_omega_n.append(lmap(env.vehicle.action["steering"], [-max_omega, max_omega], [-1, 1]))
            action_his_accel.append(env.vehicle.action["acceleration"])
            action_his_accel_n.append(lmap(env.vehicle.action["acceleration"], [-max_a, max_a], [-1, 1]))

            x_his.append(env.vehicle.position[0])
            y_his.append(env.vehicle.position[1])
            h_his.append(env.vehicle.heading)
            s_his.append(env.vehicle.speed)

            x_his_n.append(lmap(env.vehicle.position[0], [-(road_r+4), (road_r+4)], [-1, 1]))
            y_his_n.append(lmap(env.vehicle.position[1], [0, 2*(road_r+4)], [-1, 1]))
            h_his_n.append(lmap(env.vehicle.heading, [-2 * np.pi, 2*np.pi], [-1, 1]))
            s_his_n.append(lmap(env.vehicle.speed, [-max_v, max_v], [-1, 1]))
            # env.render()
            # time.sleep(0.1)
        env.close()

        tt = temp + x_road + y_road + action_his_omega + action_his_accel
        tt_n = temp_n + x_road_n + y_road_n + action_his_omega_n + action_his_accel_n
        pp = x_his + y_his + h_his + s_his
        pp_n = x_his_n + y_his_n + h_his_n + s_his_n
        lane_change = target_lane_id - lane_id
        label = labels_index[lane_change + 1, lon_operation]
        s0 = temp
        s0_n = temp_n
        ss = x_road + y_road
        ss_n = x_road_n + y_road_n
        aa = action_his_omega + action_his_accel
        aa_n = action_his_omega_n + action_his_accel_n

        s0_his.append(s0)
        s0_his_n.append(s0_n)

        ss_his.append(ss)
        ss_his_n.append(ss_n)
        aa_his.append(aa)
        aa_his_n.append(aa_n)
        tt_his.append(tt)
        tt_his_n.append(tt_n)
        pp_his.append(pp)
        pp_his_n.append(pp_n)
        label_his.append(label)

        # (z, u, s0, X, ss, road_r)
        if save_pid_data is False:
            one_data = (label, aa_n, s0_n, pp_n, ss_n, road_r)
            bf.put(one_data)

    if save_pid_data:
        np.save("label_his.npy", np.array(label_his))
        np.save("aa_his.npy", np.array(aa_his))
        np.save("aa_his_n.npy", np.array(aa_his_n))
        np.save("s0_his.npy", np.array(s0_his))
        np.save("s0_his_n.npy", np.array(s0_his_n))
        np.save("pp_his.npy", np.array(pp_his))
        np.save("pp_his_n.npy", np.array(pp_his_n))
        np.save("ss_his.npy", np.array(ss_his))
        np.save("ss_his_n.npy", np.array(ss_his_n))
        np.save("tt_his.npy", np.array(tt_his))
        np.save("tt_his_n.npy", np.array(tt_his_n))

        np.save("road_r_his.npy", np.array(road_r_his))


        print("保存了PID的数据")

def train_cl(nums=10001, save_model=False):
    loss_his = []
    for i in range(nums):
        z_his, u_his, s0_his, X_his, ss_his, cl_his, road_r_his = bf.sample(1000)
        cl_his = torch.Tensor(cl_his)
        z_his = torch.Tensor(z_his)
        feat = clen(cl_his)
        loss = clloss(feat, z_his)
        loss_his.append(loss.item())
        cl_optimizer.zero_grad()
        loss.backward()
        cl_optimizer.step()

        if i % 100 == 0:
            print("第 {} 次训练的损失是 {}".format(i, np.mean(loss_his[-100:])))
    plt.figure(1)
    plt.plot(np.arange(len(loss_his)), loss_his)
    plt.title("CL loss")
    plt.show()
    if save_model:
        torch.save(clen, 'clen_model.pkl')
        torch.save(clloss, 'clloss_model.pkl')

def eval_cl():
    z_his, u_his, s0_his, X_his, ss_his, cl_his, road_r_his = bf.sample(500)
    cl_his = torch.Tensor(cl_his)
    feat = clen(cl_his)
    X = feat.detach().numpy()
    X_embedded = TSNE(n_components=2).fit_transform(X)
    plt.figure(2)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
    plt.title("CL")
    plt.show()

def train_embedding(nums=10, save_embedding=False):
    loss_his = []
    for i in range(nums):
        z_his, u_his, s0_his, X_his, ss_his, cl_his, road_r_his = bf.sample(1000)
        cl_his = torch.Tensor(cl_his)
        z_his = torch.Tensor(z_his)
        feats = clen(cl_his)
        labels = z_his.contiguous().view(-1, 1)

        loss = 0
        for j in range(9):
            mask = torch.eq(labels, j)
            tt = feats.masked_select(mask).view(-1, 24)

            # print(feats.shape)
            loss += torch.dist(tt.detach(), embedding.weight[j, :], 2)

        em_optimizer.zero_grad()
        loss.backward()
        em_optimizer.step()

        print("第 {} 次的 loss 是 {}".format(i+1, loss.item()))

    X = embedding.weight.detach().numpy()
    if save_embedding:
        np.save("embedding.npy", X)
    X_embedded = TSNE(n_components=2).fit_transform(X)
    plt.figure(3)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
    plt.title("Embedding")
    plt.show()

def train_pp(nums=10001, save_model=False):
    re_loss_his = []
    ni_loss_his = []
    loss_his = []
    for i in range(nums):
        z_his, u_his, s0_his, X_his, ss_his, cl_his, road_r_his = bf.sample(512)
        s0_his = torch.Tensor(s0_his)
        ss_his = torch.Tensor(ss_his)
        cl_his = torch.Tensor(cl_his)
        labels_his = torch.Tensor(z_his)
        X_his = torch.Tensor(X_his)
        # embedds = embedding.weight
        embedds = torch.Tensor(em)
        pp_hat, s0_mu, ss_mu, s0_log_sigma, ss_log_sigma = clpp(s0_his, ss_his, labels_his, embedds)
        X_F = X_his.reshape(512, -1, 50)
        X_F = X_F[:, :, -1]
        pp_F = pp_hat.reshape(512, -1, 50)
        pp_F = pp_F[:, :, -1]
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_s0 = -0.5 * torch.sum(1 + s0_log_sigma - s0_mu.pow(2) - s0_log_sigma.exp())
        KLD_ss = -0.5 * torch.sum(1 + ss_log_sigma - ss_mu.pow(2) - ss_log_sigma.exp())
        loss = torch.dist(pp_hat, X_his) + 0.05 * (KLD_s0 + KLD_ss) + 2000 * torch.dist(X_F, pp_F)
        re_loss_his.append(torch.dist(X_F, pp_F).item())
        ni_loss_his.append(0.05 * (KLD_s0 + KLD_ss).item())
        loss_his.append(loss.item())
        # loss = torch.dist(pp_hat, X_his)

        pp_optimizer.zero_grad()
        loss.backward()
        pp_optimizer.step()

        if i % 100 == 0:
            print("第 {} 次的 loss 是 {}".format(i + 1, loss.item()))

    if save_model:
        torch.save(clpp, "clpp_model.pkl")

    plt.figure(4)
    plt.plot(np.arange(len(re_loss_his)), re_loss_his)
    plt.title("final state recon loss")
    # plt.figure(5)
    # plt.scatter(np.arange(len(ni_loss_his)), ni_loss_his)
    # plt.figure(6)
    # plt.scatter(np.arange(len(loss_his)), loss_his)
    plt.show()


def train_de(nums=5001, save_model=False):
    loss_his = []
    for i in range(nums):
        z_his, u_his, s0_his, X_his, ss_his, cl_his, road_r_his = bf.sample(512)
        s0_his = torch.Tensor(s0_his)
        ss_his = torch.Tensor(ss_his)
        cl_his = torch.Tensor(cl_his)
        labels_his = torch.Tensor(z_his)
        X_his = torch.Tensor(X_his)
        u_his = torch.Tensor(u_his)

        labels = labels_his.type(torch.LongTensor)
        labels = F.one_hot(labels)
        labels = labels.type(torch.float)
        skills = torch.matmul(labels, torch.Tensor(em))

        actions = clde(s0_his, ss_his, skills)
        feat = clen(torch.cat((s0_his, ss_his, actions), 1))

        # labels_his_hat = torch.argmin(torch.cdist(feat, embedding.weight.detach()), dim=1, keepdim=True)
        # labels_his_hat = torch.argmin(torch.cdist(feat, torch.Tensor(em)), dim=1)
        # re_loss = F.mse_loss(labels_his_hat, labels_his)
        re_loss = torch.dist(u_his, actions)
        norm_loss = actions.norm(dim=1).sum()
        loss = re_loss + 0.001 * norm_loss
        loss_his.append(re_loss.item())

        if i % 100 == 0:
            print("第 {} 次的 loss 是 {}".format(i + 1, loss.item()))

        de_optimizer.zero_grad()
        loss.backward()
        de_optimizer.step()

    if save_model:
        torch.save(clde, "clde_model.pkl")

    plt.figure(7)
    plt.plot(np.arange(len(loss_his)), loss_his)
    plt.title("de recon loss")
    plt.show()

# pid_collector(nums=5000, save_pid_data=True) # 用来离线收集PID数据
# train_cl(nums=2000, save_model=False)
# eval_cl()
# train_embedding(save_embedding=False)

train_pp(nums=2000, save_model=False)

train_de(save_model=False)