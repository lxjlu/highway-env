import numpy as np
import gym
import highway_env
import time
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
from nets import ReplayBuffer, CLEncoder, CLLoss, CLPP, CLDeconder, ActionNet
from sklearn.manifold import TSNE
from highway_env.utils import lmap

save_pid_data = False
load_pid = True
save_embedding = True

bf = ReplayBuffer(load_pid)
clen = CLEncoder()
clloss = CLLoss()
clpp = CLPP()
clde = CLDeconder()
actor = ActionNet()

embedding = torch.nn.Embedding(9, 24)
cl_optimizer = optim.Adam(clen.parameters(), lr=1e-3)
em_optimizer = optim.Adam(embedding.parameters(), lr=1e-3)
pp_optimizer = optim.Adam(clpp.parameters(), lr=1e-4)
de_optimizer = optim.Adam(clde.parameters(), lr=1e-4)


actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)


def pid_collector(nums=10000, save_pid_data=False):
    s0_his, ss_his, aa_his, tt_his, pp_his, label_his = [], [], [], [], [], []
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

        # positon_x = 100
        positon_x = np.random.choice(np.arange(0, env.road.network.get_lane(v_lane_id).length, 5))
        positon_y = np.random.choice(np.arange(-2, 2.1, 0.5))
        # positon_y = 0
        # heading = env.road.network.get_lane(v_lane_id).heading_at(positon_x)

        heading = np.random.choice(
            env.road.network.get_lane(v_lane_id).heading_at(positon_x) + np.arange(-np.pi / 12, np.pi / 12, 10))
        speed = np.random.choice(np.arange(0, 25, 2))
        # speed = 10

        position = env.road.network.get_lane(v_lane_id).position(positon_x, positon_y)
        inital_state = [position, heading, speed]

        env.config["v_lane_id"] = v_lane_id
        env.config["v_target_id"] = target_lane_id2
        env.config["v_x"] = positon_x
        env.config["v_y"] = positon_y
        env.config["v_h"] = heading
        env.config["v_s"] = speed
        env.config["v_target_s"] = v_target_s
        env.config["action"] = {'type': 'DiscreteMetaAction'},

        env.reset()

        p = env.vehicle.position
        i_h = env.vehicle.heading
        i_s = env.vehicle.speed
        # temp = [p[0], p[1], i_h, i_s]
        # radius*(end_phase - start_phase) * self.direction
        road_length = env.road.network.get_lane(v_lane_id).length
        temp = [lmap(p[0], [-254, 254], [-1, 1]), lmap(p[1], [0, 508], [-1, 1]),
                lmap(i_h, [-2*np.pi, np.pi], [-1, 1]), lmap(i_s, [-30, 30], [-1, 1])]
        x_road, y_road = env.vehicle.target_lane_position(p)

        action_his_omega = []
        action_his_accel = []
        action = 1
        x_his, y_his, h_his, s_his = [], [], [], []

        for _ in range(N):
            if env.vehicle.on_road is False:
                print("出去了")
                break

            env.step(action)
            action_his_omega.append(env.vehicle.action["steering"])
            action_his_accel.append(env.vehicle.action["acceleration"])
            x_his.append(lmap( env.vehicle.position[0], [-254, 254], [-1, 1]))
            y_his.append(lmap( env.vehicle.position[1], [0, 508], [-1, 1]))
            h_his.append(lmap( env.vehicle.heading, [-2*np.pi, np.pi], [-1, 1]))
            s_his.append(lmap( env.vehicle.speed, [-30, 30], [-1, 1]))
            # env.render()
            # time.sleep(0.1)
        env.close()

        tt = temp + x_road + y_road + action_his_omega + action_his_accel
        pp = x_his + y_his + h_his + s_his
        lane_change = target_lane_id - lane_id
        label = labels_index[lane_change + 1, lon_operation]
        s0 = temp
        ss = x_road + y_road
        aa = action_his_omega + action_his_accel

        s0_his.append(s0)
        ss_his.append(ss)
        aa_his.append(aa)
        tt_his.append(tt)
        pp_his.append(pp)
        label_his.append(label)
        # (z, u, s0, X, ss)
        if save_pid_data is False:
            one_data = (label, aa, s0, pp, ss)
            bf.put(one_data)

    if save_pid_data:
        np.save("label_his.npy", np.array(label_his))
        np.save("aa_his.npy", np.array(aa_his))
        np.save("s0_his.npy", np.array(s0_his))
        np.save("pp_his.npy", np.array(pp_his))
        np.save("ss_his.npy", np.array(ss_his))
        np.save("tt_his.npy", np.array(tt_his))

        # np.savetxt("label_his.csv", np.array(label_his), delimiter=",")
        # np.savetxt("aa_his.csv", np.array(aa_his), delimiter=",")
        # np.savetxt("s0_his.csv", np.array(s0_his), delimiter=",")
        # np.savetxt("pp_his.csv", np.array(pp_his), delimiter=",")
        # np.savetxt("ss_his.csv", np.array(ss_his), delimiter=",")
        # np.savetxt("tt_his.csv", np.array(tt_his), delimiter=",")
        print("保存了PID的数据")


def train_cl(nums=5001):
    loss_his = []
    for i in range(nums):
        z_his, u_his, s0_his, X_his, ss_his, cl_his = bf.sample(512)
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
    plt.show()


def eval_cl():
    z_his, u_his, s0_his, X_his, ss_his, cl_his = bf.sample(1000)
    cl_his = torch.Tensor(cl_his)
    feat = clen(cl_his)
    X = feat.detach().numpy()
    X_embedded = TSNE(n_components=2).fit_transform(X)
    plt.figure(2)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
    plt.title("CL")
    plt.show()


def train_embedding(nums=10):
    loss_his = []
    for i in range(nums):
        z_his, u_his, s0_his, X_his, ss_his, cl_his = bf.sample(1000)
        cl_his = torch.Tensor(cl_his)
        z_his = torch.Tensor(z_his)
        feats = clen(cl_his)
        labels = z_his.contiguous().view(-1, 1)

        loss = 0
        for i in range(9):
            mask = torch.eq(labels, i)
            tt = feats.masked_select(mask).view(-1, 24)

            # print(feats.shape)
            loss += torch.dist(tt.detach(), embedding.weight[i, :], 2)

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

def train_pp(nums=8001):
    for i in range(nums):
        z_his, u_his, s0_his, X_his, ss_his, cl_his = bf.sample(512)
        s0_his = torch.Tensor(s0_his)
        ss_his = torch.Tensor(ss_his)
        cl_his = torch.Tensor(cl_his)
        labels_his = torch.Tensor(z_his)
        X_his = torch.Tensor(X_his)
        embedds = embedding.weight
        pp_hat, s0_mu, ss_mu, s0_log_sigma, ss_log_sigma = clpp(s0_his, ss_his, labels_his, embedds)
        X_F = X_his.reshape(512, -1, 50)
        X_F = X_F[:, :, -1]
        pp_F = pp_hat.reshape(512, -1, 50)
        pp_F = pp_F[:, :, -1]
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_s0 = -0.5 * torch.sum(1 + s0_log_sigma - s0_mu.pow(2) - s0_log_sigma.exp())
        KLD_ss = -0.5 * torch.sum(1 + ss_log_sigma - ss_mu.pow(2) - ss_log_sigma.exp())
        loss =torch.dist(pp_hat, X_his) + 0.001 * (KLD_s0 + KLD_ss) + 2000 * torch.dist(X_F, pp_F)
        # loss = torch.dist(pp_hat, X_his)

        pp_optimizer.zero_grad()
        loss.backward()
        pp_optimizer.step()

        if i % 100 == 0:
            print("第 {} 次的 loss 是 {}".format(i + 1, loss.item()))

def train_de(nums=5001):
    for i in range(nums):
        z_his, u_his, s0_his, X_his, ss_his, cl_his = bf.sample(512)
        s0_his = torch.Tensor(s0_his)
        ss_his = torch.Tensor(ss_his)
        cl_his = torch.Tensor(cl_his)
        labels_his = torch.Tensor(z_his)
        X_his = torch.Tensor(X_his)
        u_his = torch.Tensor(u_his)

        labels = labels_his.type(torch.LongTensor)
        labels = F.one_hot(labels)
        labels = labels.type(torch.float)
        skills = torch.matmul(labels, embedding.weight.detach())

        actions = clde(s0_his, ss_his, skills)
        feat = clen(torch.cat((s0_his, ss_his, actions), 1))

        # labels_his_hat = torch.argmin(torch.cdist(feat, embedding.weight.detach()), dim=1, keepdim=True)
        labels_his_hat = torch.argmin(torch.cdist(feat, embedding.weight.detach()), dim=1)
        # re_loss = F.mse_loss(labels_his_hat, labels_his)
        re_loss = torch.dist(u_his, actions)
        norm_loss = actions.norm(dim=1).sum()
        loss = re_loss + 0.001 * norm_loss

        if i % 100 == 0:
            print("第 {} 次的 loss 是 {}".format(i + 1, loss.item()))

        de_optimizer.zero_grad()
        loss.backward()
        de_optimizer.step()

def get_first_action(s0, ss, z, em):
    hidden = embedding.weight[z]
    s_hat = clpp(s0, ss, z, em)
    u_hat = clde(s0, ss, embedding.weight[z])




def train():
    global_step = 1
    # if (save_pid_data is False) and global_step == 1:
    #     pid_collector()
    #     print("=======PID 数据收集完成=======")
    if global_step % 5000 == 1:
        print("=======开始训练对比网络=======")
        train_cl()
        print("=======对比网络训练完成=======")
        print("===============================")
        print("=======开始测试对比网络=======")
        eval_cl()
        print("========显示对比效果========")
        print("===============================")
        print("========开始训练隐藏空间========")
        train_embedding()
        print("========隐藏空间训练完成========")
        print("===============================")
        print("========开始训练预测网络========")
        train_pp()
        print("========预测网络训练完成========")
        print("===============================")
        print("========开始训练解码网络========")
        train_de()




train()
torch.save(clpp, 'pp_model.pkl')
torch.save(clde, 'de_model.pkl')

# pid_collector(nums=2000, save_pid_data=True)


# label_his, aa_his, s0_his,  pp_his, ss_his, tt_his = pid_collector(nums=1)
#
# x = pp_his.reshape(-1)[:50]
# y = pp_his.reshape(-1)[50:100]
# ss_x = ss_his.reshape(-1)[:11]
# ss_y = ss_his.reshape(-1)[11:]
# plt.plot(x, y, 'r')
# plt.plot(ss_x, ss_y, 'g')
# plt.show()
