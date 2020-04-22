# -*- coding: utf-8 -*-

import sys
from GymBase import GymBase
from config import *
from multiprocessing import Queue

from Window import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

import matplotlib.pylab as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class Comm(QThread):
    """监听外部移动信号"""
    move_signal = pyqtSignal(str)

    def __init__(self, env2ui: Queue):
        super(Comm, self).__init__()
        self.env2ui = env2ui

    def run(self):
        while True:
            self.move_signal.emit(self.env2ui.get())


class QtFigure(FigureCanvas):
    def __init__(self, width=2, height=2, dpi=100, twin=True):
        # 创建Figure
        self.twin = twin
        self.fig, self.axes = plt.subplots(
            nrows=1, ncols=1, figsize=(width, height), dpi=dpi
        )  # type: plt.Figure, plt.Axes
        self.line = None
        if twin:
            self.axes_t = self.axes.twinx()
            self.line_t = None
        # self.fig.subplots_adjust(left=0.1, right=0.9, bottom=0.2)
        self.fig.tight_layout()
        self.axes.grid()
        super(QtFigure, self).__init__(self.fig)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def _plot_(self, axes, line, *args, **kwargs):
        if not line:
            line = axes.plot(*args, **kwargs)[0]
            lines, labels = self.axes.get_legend_handles_labels()
            if self.twin:
                lines2, labels2 = self.axes_t.get_legend_handles_labels()
                if len(lines) and len(lines2):
                    self.axes.legend(lines + lines2, labels + labels2, loc='upper right')
            else:
                if len(lines):
                    self.axes.legend(lines, labels, loc='upper right')
        else:
            line.set_xdata(np.append(line.get_xdata(), args[0]))
            line.set_ydata(np.append(line.get_ydata(), args[1]))
        axes.relim()  # Recalculate limits
        axes.autoscale_view(True, True, True)  # Autoscale
        self.fig.tight_layout()
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()
        self.draw()
        return line

    def plot_(self, *args, **kwargs):
        self.line = self._plot_(self.axes, self.line, *args, **kwargs)

    def plot_t_(self, *args, **kwargs):
        self.line_t = self._plot_(self.axes_t, self.line_t, *args, **kwargs)


class UI(GymBase, Ui_MainWindow, QMainWindow):
    """移动界面"""

    def __init__(self, matrix: np.ndarray, ui2env: Queue = None, env2ui: Queue = None):
        self.app = QApplication(sys.argv)  # 创建一个应用
        GymBase.__init__(self, matrix, ui2env, env2ui)
        Ui_MainWindow.__init__(self)
        QMainWindow.__init__(self)
        self.elements = np.empty_like(matrix, dtype=np.object)
        self.setupUi(self)
        self.generate_maze()  # 生成迷宫
        # 绘图
        self.qt_figure = QtFigure()
        self.qt_figure.axes.set_xlabel('Episodes')
        self.qt_figure.axes.set_ylabel('Actions')
        self.qt_figure.axes_t.set_ylabel('Rewards')
        self.gridLayout_figure.addWidget(self.qt_figure, 0, 0)
        # 添加完整的 toolbar,增加图片放大、移动的按钮
        self.mpl_ntb = NavigationToolbar(self.qt_figure, self)
        self.gridLayout_bar.addWidget(self.mpl_ntb, 0, 0)
        # 重定义move
        self.define_move()
        if self.ui2env and self.env2ui:  # 如果有Queue信号，处理信号
            # 通讯
            self.comm = Comm(self.env2ui)
            self.comm.move_signal.connect(self.deal_with_signal)
            self.comm.start()
        else:  # 如果没有信号，操作
            # 绑定方向键
            for move_ in ACTIONS:
                getattr(self, f'pushButton_{move_}').clicked.connect(getattr(self, move_))
            # 绑定按键
            self.key_move_dict = {
                Qt.Key_Up: self.pushButton_up,
                Qt.Key_Down: self.pushButton_down,
                Qt.Key_Left: self.pushButton_left,
                Qt.Key_Right: self.pushButton_right,
            }
            self.frame.keyPressEvent = self.key_move_event
            self.frame.setFocus()
            self.pushButton_restart.clicked.connect(self.restart)
        # 计时器，更新时间
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.show_time)
        self.timer.start(1000)

    def deal_with_signal(self, x: str):
        if hasattr(self, x):
            return getattr(self, x)()
        else:
            # self.print(x)
            try:
                fun, args = x.split(':')
                if hasattr(self, fun):
                    return getattr(self, fun)(args)
                else:
                    self.print('Ignore Illegal Signal')
            except ValueError:
                self.print(f'Evaluating {x}')
                eval(x)

    def set_info(self, x):
        self.label_info.setText(self._translate(x))

    def set_param(self, x):
        self.label_param.setText(self._translate(x))

    def plot_fig(self):
        moves = self.actions[:-1]
        rewards = self.rewards[:-1]
        episodes = range(1, len(moves) + 1)
        self.qt_figure.plot_(
            episodes[-1],
            moves[-1],
            color='C0',
            label='Actions',
        )
        self.qt_figure.plot_t_(
            episodes[-1],
            rewards[-1],
            color='C1',
            label='Rewards',
        )

    def show_time(self):
        time = QDateTime().currentDateTime()  # 获取当前时间
        time_display = time.toString("yyyy-MM-dd hh:mm:ss dddd")
        self.timerLabel.setText(time_display)  # 在标签上显示时间

    @staticmethod
    def _translate(x):
        return QtCore.QCoreApplication.translate("MainWindow", x)

    def _element(self, lab):
        element = QtWidgets.QLabel(self.frame)
        element.setMinimumSize(QtCore.QSize(50, 50))
        element.setMaximumSize(QtCore.QSize(50, 50))
        element.setAlignment(QtCore.Qt.AlignCenter)
        element.setObjectName(lab)
        return element

    @staticmethod
    def _horizontal_spacer():
        return QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

    @staticmethod
    def _vertical_spacer():
        return QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

    def generate_maze(self):
        """生成迷宫"""
        for i in range(self.bound[0]):
            for j in range(self.bound[1]):
                element = self._element("label_{i}")
                self.gridLayout.addWidget(element, i + 1, j + 1, 1, 1)
                element.setStyleSheet(f"background-color:{COLOR[self.matrix[i][j]]};")
                element.setText(self._translate(f'{REWARD[self.matrix[i][j]]}'))
                element.setAlignment(QtCore.Qt.AlignCenter)
                self.elements[i][j] = element
        for i in range(1, self.bound[0] + 1):
            element = self._element("")
            self.gridLayout.addWidget(element, i, 0, 1, 1)
            element.setText(self._translate(f"x{i - 1}"))
            self.gridLayout.addItem(self._horizontal_spacer(), i, self.bound[1], 1, 1)
        for j in range(1, self.bound[1] + 1):
            element = self._element("")
            self.gridLayout.addWidget(element, 0, j, 1, 1)
            element.setText(self._translate(f"y{j - 1}"))
            self.gridLayout.addItem(self._vertical_spacer(), self.bound[0], j, 1, 1)

    def key_move_event(self, e):
        e_key = e.key()
        if e_key in self.key_move_dict:
            self.print(f'Key_Move Pressed')
            self.key_move_dict[e_key].click()
        elif e_key == Qt.Key_Escape:
            self.print('Key_Escape Pressed')
            self.close()
        elif e_key == Qt.Key_Return:
            self.print('Key_Return Pressed')
            self.restart()

    def record(self, msg, reward=None):
        super().record(msg, reward)  # 先调用父类方法
        self.statusbar.showMessage(msg)
        # self.statusbar.clearMessage()
        self.listWidget.insertItem(0, msg)
        if self.listWidget.count() > self.MAX_MSG:
            self.listWidget.takeItem(self.MAX_MSG)
        # QApplication.processEvents()
        self.moveLable.setText(self._translate(f'Action: {self.actions[-1]}'))
        self.rewardLable.setText(self._translate(f'Reward: {self.rewards[-1]}'))

    def set_state(self, pos, state):
        super().set_state(pos, state)  # 先调用父类方法
        self.elements[pos].setStyleSheet(f"background-color:{COLOR[state]};")

    def add_moves_rewards(self):
        super().add_moves_rewards()  # 先调用父类方法
        self.roundLable.setText(self._translate(f'Episode: {len(self.actions)}'))
        self.moveLable.setText(self._translate(f'Action: {self.actions[-1]}'))
        self.rewardLable.setText(self._translate(f'Reward: {self.rewards[-1]}'))
        self.plot_fig()
        self.frame.setFocus()

    def closeEvent(self, event):
        if self.env2ui and self.ui2env:
            self.ui2env.put('UI Closed')
        super().closeEvent(event)

    def run(self):
        # sys.exit(self.app.exec_())
        return self.run_().exec_()

    def run_(self):
        self.show()
        # self.showMaximized()  # 最大化展示
        if self.ui2env and self.env2ui:
            self.ui2env.put('UI Ready')
        return self.app


class DUI(UI):
    """训练DUI"""

    def __init__(self, matrix: np.ndarray, ui2env: Queue = None, env2ui: Queue = None):
        super(DUI, self).__init__(matrix, ui2env, env2ui)
        # 绘图
        self.loss_figure = QtFigure(twin=False)
        self.loss_figure.axes.set_xlabel('Steps')
        self.loss_figure.axes.set_ylabel('Losses')
        self.gridLayout_figure.addWidget(self.loss_figure, 0, 1)
        # 添加完整的 toolbar,增加图片放大、移动的按钮
        self.loss_mpl_ntb = NavigationToolbar(self.loss_figure, self)
        self.gridLayout_bar.addWidget(self.loss_mpl_ntb, 0, 1)
        self.losses = []

    def add_loss(self, x):
        # self.print(x)
        self.losses.append(float(x))
        self.plot_loss()

    def plot_loss(self):
        losses = self.losses
        episodes = range(1, len(losses) + 1)
        self.loss_figure.plot_(
            episodes[-1],
            losses[-1],
            color='C0',
            label='Losses',
        )
