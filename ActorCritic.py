# -*- coding: utf-8 -*-
import time

from config import *
from Env import DEnv
from RLBase import RLBase

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ActorCritic(RLBase):
    def __init__(self, matrix, device: torch.device, ui2env=None, env2ui=None,
                 learning_rate=LEARNING_RATE, reward_decay=REWARD_DECAY):
        super(ActorCritic, self).__init__(matrix, ui2env, env2ui, DEnv, device)
        self.lr = learning_rate
        self.gamma = reward_decay
        self.device = device

