# -*- coding: utf-8 -*-
import torch
from config import *
from multiprocessing import Queue
from GymBase import GymBase


class EnvBase(GymBase):
    """运行环境"""

    def __init__(self, matrix: np.ndarray,
                 ui2env: Queue = None, env2ui: Queue = None,
                 module=np, convert=np.array):
        super(EnvBase, self).__init__(matrix, ui2env, env2ui, module, convert)
        self.define_move()
        self.restart = self.wrapper(super().restart, 'restart')

    def wrapper(self, fun, signal, *args, **kwargs):
        """发信号包装器"""

        def wrap():
            self.send2ui(signal)
            return fun(*args, **kwargs)

        return wrap

    def send2ui(self, signal):
        if self.env2ui:
            assert self.env2ui.qsize() < 5, 'No Corresponding UI'
            self.env2ui.put(signal)

    def wait_ui(self):
        if self.env2ui and self.ui2env:
            self.print(self.ui2env.get())  # 等待UI的消息


class Env(EnvBase):
    def __init__(self, matrix: np.ndarray, ui2env: Queue = None, env2ui: Queue = None):
        super(Env, self).__init__(matrix, ui2env, env2ui)


class DEnv(EnvBase):
    """Torch版本"""

    def __init__(self, matrix: np.ndarray, device: torch.device, ui2env: Queue = None, env2ui: Queue = None):
        super(DEnv, self).__init__(matrix, ui2env, env2ui, torch, torch.tensor)
        self.device = device
        self.matrix = self.matrix.to(device)
