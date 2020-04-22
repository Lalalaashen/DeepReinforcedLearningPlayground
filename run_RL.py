# -*- coding: utf-8 -*-
"""强化学习RL"""
# %% import and define
import os
import time
import argparse
from config import *
from Env import Env
from UI import UI
from multiprocessing import Process, Queue
from QLearning import QLearning, Sarsa, SarsaLambda


class MoveUI(Process):
    """自动运行DEMO"""
    MAX_MOVES = 10
    SEC = 0.2

    def __init__(self, ui2sig: Queue = None, sig2ui: Queue = None):
        super(MoveUI, self).__init__()
        self.ui2sig = ui2sig
        self.sig2ui = sig2ui
        self.moves = 0

    def run(self):
        print(f'Action PID: {os.getpid()}')
        print(self.ui2sig.get())  # 等待UI的消息
        while self.moves < self.MAX_MOVES:
            self.sig2ui.put(ACTIONS[np.random.choice(4)])
            time.sleep(self.SEC)
            self.moves += 1
        print('Action().run() Stopped')
        self.sig2ui.put('close')


def run_ui(ui_model, ui2env, env2ui):
    ui = ui_model(MATRIX, ui2env, env2ui)
    ui.run()


def env_ui(model, *args, ui_model=UI):
    ui2env, env2ui = Queue(), Queue()
    env = model(MATRIX, *args, ui2env, env2ui)
    ui_process = Process(target=run_ui, args=(ui_model, ui2env, env2ui,))  # UI是子进程
    ui_process.daemon = True
    ui_process.start()
    print('Waiting UI to Start')
    print(env.wait_ui())
    while True:  # 循环是主进程
        getattr(env, ACTIONS[np.random.choice(4)])()
        time.sleep(0.1)


def q_base_ui(model, ui_model=UI, train_rounds=30, e_schedule=None,
              test_rounds=30, secs=0.03, mute_train=False, **kwargs):
    ui2env, env2ui = Queue(), Queue()
    p = Process(target=run_ui, args=(ui_model, ui2env, env2ui,))  # UI是子进程
    p.daemon = True
    p.start()
    if mute_train:
        m = model(MATRIX, ui2env=None, env2ui=None, **kwargs)  # type: RLBase
        if e_schedule:
            m.set_epsilon_schedule(*e_schedule)
        m.run(train_rounds, 0, train=True)
        m.set_up_env(ui2env=ui2env, env2ui=env2ui)
        m.run(test_rounds, secs, train=False, wait_start=True, wait_stop=True)
    else:
        m = model(MATRIX, ui2env=ui2env, env2ui=env2ui, **kwargs)  # type: RLBase
        if e_schedule:
            m.set_epsilon_schedule(*e_schedule)
        m.run(train_rounds, secs, train=True, wait_stop=False)
        m.run(test_rounds, secs, train=False, wait_start=False)
    return m


# %%
if __name__ == '__main__':
    print(f'Main PID: {os.getpid()}')

    args_ = AP()
    args_.s_env = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", action="store_true")
    parser.add_argument("--ui", action="store_true")
    parser.add_argument("--ui_move", action="store_true")
    parser.add_argument("--env_ui", action="store_true")
    parser.add_argument("--q_env", action="store_true")
    parser.add_argument("--q_ui", action="store_true")
    parser.add_argument("--s_env", action="store_true")
    parser.add_argument("--s_ui", action="store_true")
    parser.add_argument("--sl_env", action="store_true")
    parser.add_argument("--sl_ui", action="store_true")
    parser.add_argument("--lamb", default=0, type=float)
    parser.add_argument("--mute_train", action="store_true")
    args_ = parser.parse_args()  # will lead to unrecognized arguments error when running in console

    lamb = args_.lamb

    # %% Env Only
    if args_.env:
        env_ = Env(MATRIX)
        env_.up()
        env_.down()
        env_.left()
        env_.right()
        env_.restart()

    # %% UI Only
    if args_.ui:
        ui_ = UI(MATRIX)
        ui_.run()

    # %% UI + Action
    if args_.ui_move:
        ui2sig_, sig2ui_ = Queue(), Queue()
        ui_ = UI(MATRIX, ui2sig_, sig2ui_)  # UI是主进程
        app_ = ui_.run_()
        process_ = MoveUI(ui2sig_, sig2ui_)  # MoveUI是子进程
        process_.daemon = True
        process_.start()
        app_.exec_()
        process_.terminate()

    # %% Env + UI
    if args_.env_ui:
        env_ui(Env)

    # %% QLearning + Env
    if args_.q_env:
        q_learning_ = QLearning(MATRIX)  # QLearning是主进程
        q_learning_.run(5)

    # %% QLearning + UI
    if args_.q_ui:
        # q_learning_ = q_base_ui(QLearning, mute_train=args.mute_train)
        q_learning_ = q_base_ui(
            QLearning,
            train_rounds=10,
            e_schedule=(0.5, 0.95, 10),
            test_rounds=30,
            secs=0.03,
            mute_train=args_.mute_train,
        )

    # %% Sarsa + Env
    if args_.s_env:
        sarsa_ = Sarsa(MATRIX)  # Sarsa
        sarsa_.run(5)

    # %% Sarsa + UI
    if args_.s_ui:
        sarsa_ = q_base_ui(
            Sarsa,
            train_rounds=10,
            e_schedule=(0.5, 0.95, 10),
            test_rounds=30,
            secs=0.03,
            mute_train=args_.mute_train,
        )

    # %% Sarsa + Env
    if args_.sl_env:
        sarsa_lambda_ = SarsaLambda(MATRIX, lamb)  # QLearning是主进程
        sarsa_lambda_.run(5)

    # %% Sarsa + UI
    if args_.sl_ui:
        sarsa_lambda_ = q_base_ui(
            SarsaLambda,
            lamb=lamb,
            train_rounds=10,
            e_schedule=(0.5, 0.95, 10),
            test_rounds=30,
            secs=0.03,
            mute_train=args_.mute_train,
        )
