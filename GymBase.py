# -*- coding: utf-8 -*-
from config import *
from multiprocessing import Queue


class GymBase:
    MAX_MSG = 50

    def __init__(self, matrix: np.ndarray,
                 ui2env: Queue, env2ui: Queue,
                 module=np, convert=np.array):
        assert (matrix == State.Now).sum() == 1
        assert (matrix == State.End).sum() == 1
        self.convert = convert
        self.module = module
        self.matrix = convert(matrix)
        self.bound = matrix.shape
        self.ui2env = ui2env
        self.env2ui = env2ui
        self.start_pos = tuple(self.convert(self.module.where(self.matrix == State.Now)).flatten())
        self.end_pos = tuple(self.convert(self.module.where(self.matrix == State.End)).flatten())
        self.now_pos = self.start_pos
        self.now_pre_state = None
        # 移动历史
        self.history = []
        # self.positions = [self.now_pos]
        self.actions = [0]
        self.rewards = [0]

    def reach_end(self):
        return self.end_pos == self.now_pos

    def define_move(self):
        # 定义移动属性
        for move_, dir_ in zip(ACTIONS, STEPS):
            setattr(self, move_, self.wrapper(self.step, move_, dir_))

    def wrapper(self, fun, signal, *args, **kwargs):
        """正常包装器"""

        def wrap():
            return fun(*args, **kwargs)

        return wrap

    def check_legality(self, delta):
        """检查移动合法性"""
        next_pos = tuple(self.convert(self.now_pos) + delta)
        if all(0 <= next_pos[i] < self.bound[i] for i in range(len(delta))):
            legal = State(int(self.matrix[next_pos]))
        else:
            legal = State.Bnd
        reward = REWARD[legal]
        return legal, reward, next_pos

    @classmethod
    def print(cls, x):
        print(f'{cls.__name__} -> {x}')

    def record(self, msg, reward=None):
        self.history.insert(0, msg)
        if len(self.history) > self.MAX_MSG:
            self.history.pop(-1)
        self.print(msg)
        if reward is not None:
            self.actions[-1] += 1
            self.rewards[-1] += reward

    def step(self, delta):
        """移动"""
        delta = self.convert(delta)
        assert self.module.abs(delta).sum() == 1
        legal, reward, next_pos = self.check_legality(delta)
        if legal in {State.Emp, State.Trp, State.End}:
            old_pos, self.now_pos = self.now_pos, next_pos
            self.record(f'Moved from {old_pos} to {next_pos}', reward)
            # self.positions.append(self.now_pos)
            self.set_state(old_pos, self.now_pre_state if self.now_pre_state else State.Emp)
            self.now_pre_state = self.matrix[next_pos]
            self.set_state(next_pos, State.Now)
        else:
            self.record(f'Illegal move from {self.now_pos} to {next_pos}', reward)
        return reward

    def restart(self):
        """重启"""
        self.record(f'{"Game Ends. " if self.now_pos == self.end_pos else ""}'
                    f'Restart. Moved from {self.now_pos} to {self.start_pos}')
        self.set_state(self.now_pos, State.Emp)
        self.set_state(self.end_pos, State.End)
        self.set_state(self.start_pos, State.Now)
        # 回归起点
        self.now_pos = self.start_pos
        self.now_pre_state = None
        self.add_moves_rewards()

    def set_state(self, pos, state):
        self.matrix[pos] = state

    def add_moves_rewards(self):
        self.actions.append(0)
        self.rewards.append(0)

    def up(self):
        """后续被重载"""

    def down(self):
        """后续被重载"""

    def left(self):
        """后续被重载"""

    def right(self):
        """后续被重载"""
