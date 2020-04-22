# -*- coding: utf-8 -*-
"""深度强化学习RL"""
# %% import and define
import torch
import argparse
from config import *
from run_RL import env_ui, q_base_ui
from Env import DEnv
from UI import DUI
from DQLearning import DQLearning
from DPolicyGradient import DVanillaPolicyGradient
from multiprocessing import Process, Queue

# %%
if __name__ == '__main__':
    args_ = AP()
    args_.dvpg = True
    args_.device = 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cpu', type=str)

    parser.add_argument("--env", action="store_true")
    parser.add_argument("--env_ui", action="store_true")

    parser.add_argument("--dq", action="store_true")
    parser.add_argument("--dq_ui", action="store_true")
    parser.add_argument("--double", action="store_true")

    parser.add_argument("--dvpg", action="store_true")
    parser.add_argument("--dvpg_ui", action="store_true")
    args_ = parser.parse_args()  # will lead to unrecognized arguments error when running in console

    device_ = torch.device(args_.device)
    if args_.env:
        env_ = DEnv(MATRIX, device_)
        env_.up()
        env_.down()
        env_.left()
        env_.right()
        env_.restart()

    # %% Env + UI
    if args_.env_ui:
        env_ui(DEnv, device_, ui_model=DUI)

    if args_.dq:
        dq_learning_ = DQLearning(
            MATRIX, device_, args_.double
        )
        dq_learning_.run(episodes=100, secs=None, train=True)

    if args_.dq_ui:
        dq_learning_ = q_base_ui(
            DQLearning,
            device=device_,
            double=args_.double,
            ui_model=DUI,
            train_rounds=50,
            e_schedule=(0.5, 0.8, 50),
            test_rounds=10,
            secs=0.05,
        )

    if args_.dvpg:
        dvpg_ = DVanillaPolicyGradient(
            MATRIX, device_,
        )
        dvpg_.run(episodes=100, secs=None, train=True)

    if args_.dvpg_ui:
        dvpg_ = q_base_ui(
            DVanillaPolicyGradient,
            device=device_,
            ui_model=DUI,
            train_rounds=100,
            test_rounds=10,
            secs=0.02,
            # secs=0.1,
        )
