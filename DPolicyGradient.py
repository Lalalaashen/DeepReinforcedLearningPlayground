# -*- coding: utf-8 -*-
import time

from config import *
from Env import DEnv
from RLBase import RLBase

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class PNet(nn.Module):
    def __init__(self, n_features, n_actions):
        super(PNet, self).__init__()
        hidden = 64
        self.seq = nn.Sequential(
            nn.Linear(n_features, hidden, True),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions, True),
            nn.Softmax(1),
        )

    def forward(self, x):
        return self.seq(x)


D_LEARNING_RATE = 0.01


class DVanillaPolicyGradient(RLBase):
    """深度Policy Gradient算法 On-policy"""

    def __init__(self, matrix, device: torch.device, ui2env=None, env2ui=None,
                 learning_rate=D_LEARNING_RATE, reward_decay=REWARD_DECAY):
        super(DVanillaPolicyGradient, self).__init__(matrix, ui2env, env2ui, DEnv, device)
        self.lr = learning_rate
        self.gamma = reward_decay
        self.device = device
        self.net = PNet(2, self.n_action).to(device)  # type: PNet.forward
        # self.optimizer = optim.RMSprop(self.net.parameters())
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.ep_obs = []
        self.ep_as = []
        self.ep_rs = []

    def to_tensor(self, x):
        return torch.tensor(x, dtype=torch.float, device=self.device).unsqueeze(0)

    def set_ui_param(self):
        self.env.send2ui(f'set_param:lr = {self.lr:.3f}, gamma = {self.gamma:.3f}')

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def get_next_action(self, pos):
        """根据概率选取action"""
        self.net.eval()
        prob_weights = self.net(self.to_tensor(pos)).detach().cpu().numpy()  # type: np.ndarray
        action_num = np.random.choice(self.n_action, p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action_num

    def take_action(self):
        """进行移动决策"""
        pos, action_num, next_pos, reward = self._step()
        self.store_transition(pos, action_num, {
            0: 10,
            -1: 5,
            -2: -1000,
            20: 20
        }[reward])  # 为了改善默认走路状态为0的情况
        # 更新对应的QTable值
        return self.check_continue()

    def loss(self, probs, acts, vt):
        """Average reward per time-step"""
        # acts = torch.zeros(acts.size()[0], self.n_action, dtype=torch.long).scatter_(1, acts, 1)  # ->one hot
        neg_log_prob = F.nll_loss(probs, acts.flatten())
        return torch.mean(neg_log_prob * vt)

    def _discount_and_norm_rewards(self):
        """discount episode rewards"""
        discounted_ep_rs = torch.zeros(len(self.ep_rs), device=self.device)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        # normalize episode rewards
        discounted_ep_rs -= torch.mean(discounted_ep_rs)
        discounted_ep_rs /= torch.std(discounted_ep_rs)
        return discounted_ep_rs.reshape((-1, 1))

    def optimize_model(self):
        """模型优化"""
        self.net.train()
        vt = self._discount_and_norm_rewards()
        obs = torch.tensor(self.ep_obs, dtype=torch.float, device=self.device)
        acts = torch.tensor(self.ep_as, device=self.device)
        probs = self.net(obs)
        loss = self.loss(probs, acts, vt)
        # 模型优化
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # 清空历史数据
        self.ep_obs.clear()
        self.ep_as.clear()
        self.ep_rs.clear()
        return loss.detach().cpu().numpy()

    def run(self, episodes, secs, train, wait_start=True, wait_stop=True):
        """"""
        if wait_start:
            self.print('Waiting UI to Start')
            self.env.wait_ui()
        self.set_ui_info(train)
        # run
        steps = 0
        running_reward = None
        self.env.send2ui("self.loss_figure.axes.set_xlabel('Episodes')")
        for ep in range(episodes):
            print(f'Episode {ep}')
            self.set_ui_param()
            # PG算法需要跑完一整个episode
            while self.take_action():
                steps += 1
                if secs:
                    time.sleep(secs)
            if train:
                # episode结束，汇总学习
                ep_rs_sum = sum(self.ep_rs)
                if running_reward is None:
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                self.print(f"Episode: {ep}, Reward: {running_reward:.2f}")
                loss = self.optimize_model()
                if loss is not None:
                    self.env.send2ui(f'add_loss:{loss}')

        if wait_stop:
            self.print('Waiting UI to Stop')
            self.env.wait_ui()
