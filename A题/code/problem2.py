# -*- coding: utf-8 -*-
"""
问题2: 同尺寸定日镜场优化设计 (额定功率 60MW, 最大化单位面积功率)
径向交错布局 + 差分进化(DE)优化, 解析代理评估
参数: [xt, yt, W, H, h, r0, ds]  (dr = sqrt(3)/2 * ds 六角密排)
"""
import os
import time
import numpy as np
from scipy.optimize import differential_evolution
from layout import generate_layout, generate_layout_origin, fast_evaluate, sb_model

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P_RATED = 60000.0   # 额定功率 60MW -> kW

FULL_TIMES = [(mi, st) for mi in range(12) for st in [9.0, 10.5, 12.0, 13.5, 15.0]]

BOUNDS = [
    (-200, 200),   # xt
    (-200, 200),   # yt
    (2.0, 8.0),    # W
    (2.0, 8.0),    # H
    (2.0, 6.0),    # h
    (30.0, 100.0), # r0 (原点内圈起始)
    (7.0, 30.0),   # ds
]


def objective(x, penalty_scale=1.0):
    xt, yt, W, H, h, r0, ds = x
    # W >= H
    if W < H:
        W, H = H, W
    # h >= H/2 (镜面不触地)
    h = max(h, H / 2.0)
    if h > 6.0:
        return 100.0
    # 间距约束 ds > W+5
    if ds < W + 5.0:
        return 100.0
    dr = np.sqrt(3) / 2.0 * ds
    coords = generate_layout_origin(r0, dr, ds, (xt, yt))
    if coords.shape[0] < 50 or coords.shape[0] > 4000:
        return 100.0
    unit_power, E_kw, annual_eta, N, _, _ = fast_evaluate(
        coords, (xt, yt), W, H, h, times=FULL_TIMES, sb=sb_model(ds, W))
    # 罚: 功率不足
    shortfall = max(0.0, P_RATED - E_kw)
    f = unit_power - penalty_scale * shortfall / 10000.0
    return -f   # DE 最小化 -> 取负


def run_optimization(popsize=12, maxiter=60, seed=1, workers=1):
    t0 = time.time()
    res = differential_evolution(
        objective, BOUNDS, popsize=popsize, maxiter=maxiter, seed=seed,
        tol=1e-4, polish=True, disp=True, workers=workers,
        updating='immediate')
    print(f'优化用时 {time.time()-t0:.1f}s')
    return res


def decode(res_x):
    xt, yt, W, H, h, r0, ds = res_x
    if W < H:
        W, H = H, W
    h = max(h, H / 2.0)
    dr = np.sqrt(3) / 2.0 * ds
    coords = generate_layout(xt, yt, r0, dr, ds)
    return dict(xt=xt, yt=yt, W=W, H=H, h=h, r0=r0, dr=dr, ds=ds, coords=coords)


if __name__ == '__main__':
    res = run_optimization()
    print('\n最优参数:')
    print('  f(负目标)=', res.fun)
    d = decode(res.x)
    print(f'  塔位=({d["xt"]:.2f},{d["yt"]:.2f})  W={d["W"]:.3f}  H={d["H"]:.3f}  h={d["h"]:.3f}')
    print(f'  r0={d["r0"]:.2f}  dr={d["dr"]:.3f}  ds={d["ds"]:.3f}  N={d["coords"].shape[0]}')
    unit, E, eta, N = fast_evaluate(d['coords'], (d['xt'], d['yt']), d['W'], d['H'], d['h'])
    print(f'  代理评估: unit={unit:.5f} kW/m2  E={E/1000:.3f} MW  eta={eta:.5f}')
    np.save(os.path.join(DATA_DIR, 'output', 'problem2_best.npy'), res.x, allow_pickle=False)
    np.save(os.path.join(DATA_DIR, 'output', 'problem2_coords.npy'), d['coords'])
    print('已保存最优参数与坐标')
