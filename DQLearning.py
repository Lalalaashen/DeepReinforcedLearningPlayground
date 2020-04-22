# -*- coding: utf-8 -*-
# todo double DQN;
# todo Prioritized Experience Replay (DQN);
# todo Dueling DQN;
import time
import random
from collections import namedtuple

from config import *
from Env import DEnv
from RLBase import QBase

import torch
import torch.nn as nn
import torch.optim as optim

# PPO很容易过早收敛
Transition = namedtuple(
    typename='Transition',
    field_names=(
        'pos',
        'action_num',
        'next_pos',
        'reward',
    ),
)


class ReplayMemory:
    """记忆库"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity  # 会重新覆盖

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class QNet(nn.Module):
    """QNet"""

    def __init__(self, n_features, n_actions):
        super(QNet, self).__init__()
        hidden = 20
        self.seq = nn.Sequential(
            nn.Linear(n_features, hidden, True),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions, True),
        )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        return self.seq(x)


class DQLearning(QBase):
    """深度Q Learning"""

    def __init__(self, matrix, device: torch.device, double=False, ui2env=None, env2ui=None,
                 learning_rate=LEARNING_RATE, reward_decay=REWARD_DECAY, e_greedy=E_GREEDY,
                 update_target_steps=UPDATE_TARGET_STEPS, memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE, ):
        super(DQLearning, self).__init__(matrix, ui2env, env2ui, learning_rate, reward_decay, e_greedy,
                                         DEnv, device)
        self.device = device
        self.double = double
        self.policy_net = QNet(2, self.n_action).to(device)
        self.target_net = QNet(2, self.n_action).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.loss = nn.MSELoss()
        # self.loss = nn.SmoothL1Loss()
        self.memory = ReplayMemory(memory_size)
        self.update_target_steps = update_target_steps
        self.batch_size = batch_size

    def update_target_net(self):
        """更新target_net"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def to_tensor(self, x):
        return torch.tensor(x, dtype=torch.float, device=self.device).unsqueeze(0)
        # return torch.tensor(x, dtype=torch.float, device=self.device).view(1, 1)

    def get_next_action(self, pos):
        """Epsilon-Greedy"""
        if np.random.random() < self.epsilon:
            self.policy_net.eval()
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(self.to_tensor(pos)).max(1).indices.view(1, 1)
                # return np.random.choice(np.where(qs == qs[np.argmax(qs)])[0])  # 防止q值相同
        else:
            return torch.tensor([[random.randrange(self.n_action)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation).
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        next_state_batch = torch.cat(batch.next_pos)
        state_batch = torch.cat(batch.pos)
        action_batch = torch.cat(batch.action_num)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken.
        # These are the actions which would've been taken for each batch state according to policy_net
        self.policy_net.train()
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = self.target_net(next_state_batch).max(1).values.detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma + reward_batch).unsqueeze(1)

        loss = self.loss(state_action_values, expected_state_action_values)

        # 模型优化
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def take_action(self):
        """进行移动决策"""
        pos, action_num, next_pos, reward = self._step()
        self.memory.push(
            self.to_tensor(pos),
            action_num,
            self.to_tensor(next_pos),
            self.to_tensor(reward),
        )
        return self.check_continue()

    def run(self, episodes, secs=None, train=True, wait_start=True, wait_stop=True):
        if wait_start:
            self.print('Waiting UI to Start')
            self.env.wait_ui()
        self.set_ui_info(train)
        # run
        steps = 0
        for ep in range(episodes):
            print(f'Episode {ep}')
            self.set_ui_param()
            while self.take_action():
                steps += 1
                if train:
                    if steps % self.update_target_steps == 0:
                        self.print(f'UPDATE TARGET_NET Step = {steps}')
                        self.update_target_net()
                    loss = self.optimize_model()
                    if loss is not None:
                        self.env.send2ui(f'add_loss:{loss}')
                if secs:
                    time.sleep(secs)
            self.schedule_epsilon()
        if wait_stop:
            self.print('Waiting UI to Stop')
            self.env.wait_ui()
