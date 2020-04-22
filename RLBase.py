# -*- coding: utf-8 -*-
from config import *
from Env import Env


class RLBase:
    def __init__(self, matrix, ui2env, env2ui, env_type=Env, *args):
        self.env = env_type(matrix, *args, ui2env, env2ui)
        self.n_action = len(ACTIONS)

    def check_continue(self):
        """一个回合是否结束"""
        reach_end = self.env.reach_end()
        if reach_end:
            self.env.restart()
        return not reach_end

    def _step(self, action_num=None):
        pos = self.env.now_pos
        if action_num is None:
            action_num = self.get_next_action(pos)  # 下一步操作
        reward = getattr(self.env, ACTIONS[action_num])()
        next_pos = self.env.now_pos
        return pos, action_num, next_pos, reward

    def set_up_env(self, **kwargs):
        for key, item in kwargs.items():
            setattr(self.env, key, item)

    @classmethod
    def print(cls, x):
        print(f'{cls.__name__} -> {x}')

    def set_ui_info(self, train):
        self.env.send2ui(f'set_info:{self.__class__.__name__} (train = {train})')

    def set_ui_param(self):
        self.env.send2ui(f'set_param:lr = {self.lr:.3f}, gamma = {self.gamma:.3f}, epsilon = {self.epsilon:.3f}')

    def get_next_action(self, *args, **kwargs):
        """"""

    def take_action(self, *args, **kwargs):
        """"""

    def run(self, episodes, secs, train, wait_start=True, wait_stop=True):
        """"""


class QBase(RLBase):
    def __init__(self, matrix, ui2env, env2ui, learning_rate, reward_decay, e_greedy, env_type=Env, *args):
        super().__init__(matrix, ui2env, env2ui, env_type, *args)
        # 学习率
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.Q_table = np.zeros(tuple(self.env.bound) + (self.n_action,))

    def get_next_action(self, action):
        """Epsilon-Greedy"""
        if np.random.random() < self.epsilon:
            qs = self.Q_table[action]
            return np.random.choice(np.where(qs == qs[np.argmax(qs)])[0])  # 防止q值相同
        else:
            return np.random.choice(self.n_action)

    def schedule_epsilon(self):
        if hasattr(self, '_epsilon_schedule'):
            self._epsilon_position = min(self._epsilon_position + 1, len(self._epsilon_schedule) - 1)
            self.epsilon = self._epsilon_schedule[self._epsilon_position]

    def set_epsilon_schedule(self, start, end, episodes, dtype='linear'):
        if dtype == 'linear':
            self._epsilon_schedule = np.linspace(start, end, episodes - 1)
        elif dtype == 'exp':
            b = episodes
            self._epsilon_schedule = (np.exp(np.linspace(0, b, episodes - 1)) - 1) * (end - start) \
                                     / (np.exp(b) - np.exp(0)) + start
        else:
            raise NotImplementedError
        self._epsilon_position = 0
        self.epsilon = self._epsilon_schedule[self._epsilon_position]
