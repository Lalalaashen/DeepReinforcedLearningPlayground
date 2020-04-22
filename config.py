# -*- coding: utf-8 -*-
from enum import IntEnum, unique
import numpy as np

# Q
# LEARNING_RATE = 0.01
# REWARD_DECAY = 0.9
# E_GREEDY = 0.9
LEARNING_RATE = 0.1
REWARD_DECAY = 0.9
E_GREEDY = 0.8

# DQN
UPDATE_TARGET_STEPS = 100
MEMORY_SIZE = 500
BATCH_SIZE = 32


@unique
class State(IntEnum):
    Now = 0  # 开始
    Emp = 1  # 空
    Occ = 2  # 被占据
    Trp = 3  # 陷阱
    End = 4  # 结束
    Bnd = -1


COLOR = {
    State.Now: 'yellow',
    State.Emp: 'green',
    State.Occ: 'red',
    State.Trp: 'grey',
    State.End: 'orange',
    State.Bnd: 'black',
}

REWARD = {
    State.Now: 0,
    State.Emp: 0,
    State.Occ: -2,
    State.Trp: -1,
    State.End: 20,
    State.Bnd: -2,
}

MATRIX = np.array([
    [State.Now, State.Trp, State.Emp, State.Occ, State.Occ],
    [State.Emp, State.Occ, State.Emp, State.Emp, State.Trp],
    [State.Emp, State.Emp, State.Occ, State.Trp, State.End],
    [State.Occ, State.Emp, State.Emp, State.Emp, State.Occ],
])

ACTIONS = ['up', 'down', 'left', 'right']
STEPS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SEED = None
np.random.seed(SEED)


class AP:
    def __getattr__(self, item):
        return self.__dict__[item] if item in self.__dict__ else False
