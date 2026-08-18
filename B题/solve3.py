# -*- coding: utf-8 -*-
"""
问题 3: 南北2海里 × 东西4海里, 中心水深110m, 西深东浅, 坡度1.5°, 开角120°
设计最短总长度(最少测线数)、完全覆盖、重叠率10%~20% 的测线
"""
import numpy as np

NM = 1852.0
theta, alpha, D0 = 120.0, 1.5, 110.0
h = np.radians(theta / 2)
a = np.radians(alpha)
sh, ch = np.sin(h), np.cos(h)
tan_a = np.tan(a)

# 测线沿南北(等高线), β=90°, 视坡度=α
# 坡下侧在西(深), 坡上侧在东(浅)
def D(x):    return D0 - x * tan_a          # x 向东为正, 东浅
def w_shallow(x): return D(x) * sh / (ch + sh * tan_a)   # 东侧半宽
def w_deep(x):    return D(x) * sh / (ch - sh * tan_a)   # 西侧半宽
def W(x):    return w_shallow(x) + w_deep(x)

L_east = 2 * NM   # 东边界 x=+3704
L_west = -2 * NM  # 西边界 x=-3704

# 第一条测线: 覆盖西边界 x1 - w_deep(x1) = L_west
# 线性方程: x1 - (D0 - x1*tan_a)*k_deep = L_west, k_deep = sh/(ch-sh*tan_a)
k_deep = sh / (ch - sh * tan_a)
x1 = (L_west + D0 * k_deep) / (1 + tan_a * k_deep)

# 递推: 每条间距取使重叠率恰好=10%(最大间距, 最少测线数)
eta_min = 0.10
xs = [x1]
def find_next(xi, eta_target=eta_min):
    """解下一条测线位置 x_{i+1}, 使重叠率=eta_target(平均覆盖宽度定义)"""
    from scipy.optimize import brentq
    # 重叠率 η = 1 - d / ((W(xi)+W(xj))/2), d = xj - xi
    # f(d) = 1 - d / avg = eta_target
    def f(d):
        xj = xi + d
        avg = (W(xi) + W(xj)) / 2
        return (1 - d / avg) - eta_target
    # 区间: d ∈ (0, 0.95*W(xi)]
    dmax = 0.95 * W(xi)
    return xi + brentq(f, 1e-6, dmax)

while xs[-1] + w_shallow(xs[-1]) < L_east:
    xs.append(find_next(xs[-1]))
xs = np.array(xs)

N = len(xs)
print(f"最少测线数 N = {N}")
print(f"总长度 = {N} × 2 海里 = {N*2:.2f} 海里")
print()
print(f"{'序号':>4} {'x位置/m':>10} {'x/海里':>8} {'水深/m':>8} {'W/m':>9} {'覆盖东缘/m':>12}")
for i, x in enumerate(xs):
    east_edge = x + w_shallow(x)
    west_edge = x - w_deep(x)
    print(f"{i+1:4d} {x:10.2f} {x/NM:8.3f} {D(x):8.2f} {W(x):9.2f} {east_edge:12.2f}")

# 验证重叠率与覆盖
print()
print("验证: 相邻测线重叠率(平均宽度定义) 与 实际重叠")
print(f"{'对':>6} {'间距/m':>10} {'η(平均)/%':>10} {'η(实际)/%':>10}")
eta_actual_list = []
for i in range(N - 1):
    d = xs[i+1] - xs[i]
    avg = (W(xs[i]) + W(xs[i+1])) / 2
    eta_avg = 1 - d / avg
    # 实际重叠宽度
    overlap = w_shallow(xs[i]) + w_deep(xs[i+1]) - d
    eta_actual = overlap / avg
    eta_actual_list.append(eta_actual)
    print(f"{i+1:>4}-{i+2:<4} {d:10.2f} {eta_avg*100:10.2f} {eta_actual*100:10.2f}")

print()
print(f"覆盖西边界: {xs[0] - w_deep(xs[0]):.2f} (目标 {L_west:.2f})")
print(f"覆盖东边界: {xs[-1] + w_shallow(xs[-1]):.2f} (目标 {L_east:.2f})")
print(f"重叠率范围(平均): {min(1- (xs[i+1]-xs[i])/((W(xs[i])+W(xs[i+1]))/2) for i in range(N-1))*100:.2f}% ~ {max(1- (xs[i+1]-xs[i])/((W(xs[i])+W(xs[i+1]))/2) for i in range(N-1))*100:.2f}%")
