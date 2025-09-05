import time
import collections

# ---------------- Plot wrapper ----------------
class RollingPlot:
    """維護一個折線圖的滾動視窗資料（長度 = window_s）。支援第二條線（黃色）。"""
    def __init__(self, ax, title, y_label, y_min, y_max, window_s, hop_s, with_second=False, second_label=None):
        self.ax = ax
        self.window_s = window_s
        self.hop_s = hop_s
        self.max_points = max(2, int(window_s / hop_s))
        self.xs = collections.deque(maxlen=self.max_points)
        self.ys = collections.deque(maxlen=self.max_points)
        (self.line,) = self.ax.plot([], [], linewidth=1.6)
        self.line2 = None
        self.ys2 = []
        if with_second:
            (self.line2,) = self.ax.plot([], [], 'y', linewidth=1.6)  # 黃色線
            self.ys2 = collections.deque(maxlen=self.max_points)
            if second_label:
                self.line2.set_label(second_label)
                self.ax.legend(loc="upper right", frameon=False, fontsize=9)
        self.ax.set_title(title)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel(y_label)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_xlim(0, window_s)
        self._t0 = time.time()

    def update(self, y, y2=None):
        import numpy as np
        t = time.time() - self._t0
        self.xs.append(t)
        self.ys.append(y)
        if self.line2 is not None and y2 is not None:
            self.ys2.append(y2)
        # 顯示最近 window_s 秒
        x0 = self.xs[-1] - self.window_s if len(self.xs) > 0 else 0.0
        xs = np.array(self.xs) - max(x0, 0.0)
        ys = np.array(self.ys)
        self.line.set_data(xs, ys)
        if self.line2 is not None and self.ys2 is not None and len(self.ys2) > 0:
            ys2 = np.array(self.ys2)
            self.line2.set_data(xs[-len(ys2):], ys2)  # 對齊長度
        self.ax.set_xlim(0, self.window_s)