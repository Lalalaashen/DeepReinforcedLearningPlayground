# -*- coding: utf-8 -*-
"""QLearning离散state，离散的action"""
import time
from config import *
from RLBase import QBase


class QLearning(QBase):
    """QLearning"""

    def __init__(self, matrix, ui2env=None, env2ui=None,
                 learning_rate=LEARNING_RATE, reward_decay=REWARD_DECAY, e_greedy=E_GREEDY):
        super(QLearning, self).__init__(matrix, ui2env, env2ui, learning_rate, reward_decay, e_greedy)

    def take_action(self, train=True):
        """进行移动决策"""
        pos, action_num, next_pos, reward = self._step()
        # 更新对应的QTable值
        if train:
            q_predict = self.Q_table[pos][action_num]
            q_target = reward + self.gamma * self.Q_table[next_pos].max()  # 新位置可能的最大Reward估计
            self.Q_table[pos][action_num] += self.lr * (q_target - q_predict)
        return self.check_continue()

    def run(self, episodes, secs=None, train=True, wait_start=True, wait_stop=True):
        if wait_start:
            self.print('Waiting UI to Start')
            self.env.wait_ui()
        self.set_ui_info(train)
        for ep in range(episodes):
            print(f'Episode {ep}')
            self.set_ui_param()
            while self.take_action(train):
                if secs:
                    time.sleep(secs)
            self.schedule_epsilon()
        if wait_stop:
            self.print('Waiting UI to Stop')
            self.env.wait_ui()


class Sarsa(QBase):
    """Sarsa"""

    def __init__(self, matrix, ui2env=None, env2ui=None,
                 learning_rate=LEARNING_RATE, reward_decay=REWARD_DECAY, e_greedy=E_GREEDY):
        super(Sarsa, self).__init__(matrix, ui2env, env2ui, learning_rate, reward_decay, e_greedy)

    def take_action(self, train=True, action_num=None):
        """进行移动决策"""
        pos, action_num, next_pos, reward = self._step(action_num)
        # 更新对应的QTable值
        if train:
            next_action_num = self.get_next_action(next_pos)  # 下一步操作
            q_predict = self.Q_table[pos][action_num]
            q_target = reward + self.gamma * self.Q_table[next_pos][next_action_num]  # 基于新位置选择的Reward估计
            self.Q_table[pos][action_num] += self.lr * (q_target - q_predict)
        else:
            next_action_num = None
        return self.check_continue(), next_action_num

    def run(self, episodes, secs=None, train=True, wait_start=True, wait_stop=True):
        if wait_start:
            self.print('Waiting UI to Start')
            self.env.wait_ui()
        self.set_ui_info(train)
        for ep in range(episodes):
            print(f'Episode {ep}')
            self.set_ui_param()
            cont, action_num = True, None
            while cont:
                cont, action_num = self.take_action(train, action_num)
                if secs:
                    time.sleep(secs)
            self.schedule_epsilon()
        if wait_stop:
            self.print('Waiting UI to Stop')
            self.env.wait_ui()


class SarsaLambda(Sarsa):
    """SarsaLambda"""

    def __init__(self, matrix, lamb, ui2env=None, env2ui=None,
                 learning_rate=LEARNING_RATE, reward_decay=REWARD_DECAY, e_greedy=E_GREEDY):
        super(SarsaLambda, self).__init__(matrix, ui2env, env2ui, learning_rate, reward_decay, e_greedy)
        assert 0 <= lamb <= 1
        self.E = np.zeros(tuple(self.env.bound) + (self.n_action,))
        self.lamb = lamb
        self.print(f'lamb = {lamb}')

    def check_continue(self):
        """一个回合是否结束"""
        reach_end = not super().check_continue()
        if reach_end:
            # 回合结束，E清空
            self.E = np.zeros(tuple(self.env.bound) + (self.n_action,))
        return not reach_end

    def set_ui_param(self):
        self.env.send2ui(f'set_param:lambda = {self.lamb:.3f}, '
                         f'lr = {self.lr:.3f}, gamma = {self.gamma:.3f}, epsilon = {self.epsilon:.3f}')

    def take_action(self, train=True, action_num=None):
        """进行移动决策"""
        pos, action_num, next_pos, reward = self._step(action_num)
        # 更新对应的QTable值
        if train:
            next_action_num = self.get_next_action(next_pos)  # 下一步操作
            q_predict = self.Q_table[pos][action_num]
            q_target = reward + self.gamma * self.Q_table[next_pos][next_action_num]  # 基于新位置选择的Reward估计
            delta = q_target - q_predict
            # 更新之前的所有步
            self.E[pos][action_num] += 1
            self.Q_table += self.lr * delta * self.E
            self.E = self.gamma * self.lamb * self.E
        else:
            next_action_num = None
        return self.check_continue(), next_action_num
