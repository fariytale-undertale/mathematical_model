# -*- coding: utf-8 -*-
"""
问题3: 异尺寸/异安装高度定日镜场优化 (额定功率 60MW)
径向交错布局 + 尺寸随径向距离线性变化 + DE 优化
参数: [xt, yt, r0, ds, W_in, W_out, H_in, H_out, h_in, h_out]
"""
import os
import time
import numpy as np
from scipy.optimize import differential_evolution
from layout import generate_layout, fast_evaluate_var, sb_model

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P_RATED = 60000.0

BOUNDS = [
    (-200, 200),   # xt
    (-200, 200),   # yt
    (100.0, 140.0),# r0
    (7.0, 30.0),   # ds
    (2.0, 8.0),    # W_in
    (2.0, 8.0),    # W_out
    (2.0, 8.0),    # H_in
    (2.0, 8.0),    # H_out
    (2.0, 6.0),    # h_in
    (2.0, 6.0),    # h_out
]


def build_sizes(coords, xt, yt, r0, W_in, W_out, H_in, H_out, h_in, h_out):
    r = np.sqrt((coords[:, 0] - xt) ** 2 + (coords[:, 1] - yt) ** 2)
    rmax = r.max() if r.size else 350.0
    t = np.clip((r - r0) / max(rmax - r0, 1e-9), 0.0, 1.0)
    W = W_in + (W_out - W_in) * t
    H = H_in + (H_out - H_in) * t
    h = h_in + (h_out - h_in) * t
    return W, H, h


def objective(x, penalty_scale=1.0):
    xt, yt, r0, ds, W_in, W_out, H_in, H_out, h_in, h_out = x
    # W >= H
    if W_in < H_in:
        W_in, H_in = H_in, W_in
    if W_out < H_out:
        W_out, H_out = H_out, W_out
    h_in = max(h_in, H_in / 2.0)
    h_out = max(h_out, H_out / 2.0)
    if h_in > 6.0 or h_out > 6.0:
        return 100.0
    Wmax = max(W_in, W_out)
    if ds < Wmax + 5.0:
        return 100.0
    dr = np.sqrt(3) / 2.0 * ds
    coords = generate_layout(xt, yt, r0, dr, ds)
    if coords.shape[0] < 50 or coords.shape[0] > 6000:
        return 100.0
    W, H, h = build_sizes(coords, xt, yt, r0, W_in, W_out, H_in, H_out, h_in, h_out)
    sb = sb_model(ds, Wmax)
    unit_power, E_kw, annual_eta, N, _, _ = fast_evaluate_var(coords, (xt, yt), W, H, h, sb=sb)
    shortfall = max(0.0, P_RATED - E_kw)
    f = unit_power - penalty_scale * shortfall / 10000.0
    return -f


def decode(res_x):
    xt, yt, r0, ds, W_in, W_out, H_in, H_out, h_in, h_out = res_x
    if W_in < H_in:
        W_in, H_in = H_in, W_in
    if W_out < H_out:
        W_out, H_out = H_out, W_out
    h_in = max(h_in, H_in / 2.0)
    h_out = max(h_out, H_out / 2.0)
    dr = np.sqrt(3) / 2.0 * ds
    coords = generate_layout(xt, yt, r0, dr, ds)
    W, H, h = build_sizes(coords, xt, yt, r0, W_in, W_out, H_in, H_out, h_in, h_out)
    return dict(xt=xt, yt=yt, r0=r0, dr=dr, ds=ds,
                W_in=W_in, W_out=W_out, H_in=H_in, H_out=H_out,
                h_in=h_in, h_out=h_out, coords=coords, W=W, H=H, h=h)


if __name__ == '__main__':
    t0 = time.time()
    res = differential_evolution(objective, BOUNDS, popsize=8, maxiter=40, seed=7,
                                 tol=1e-4, polish=True, disp=True, updating='immediate')
    print(f'优化用时 {time.time()-t0:.1f}s')
    d = decode(res.x)
    print(f'塔位=({d["xt"]:.2f},{d["yt"]:.2f}) r0={d["r0"]:.2f} ds={d["ds"]:.3f} N={d["coords"].shape[0]}')
    print(f'W_in={d["W_in"]:.3f} W_out={d["W_out"]:.3f} H_in={d["H_in"]:.3f} H_out={d["H_out"]:.3f}')
    print(f'h_in={d["h_in"]:.3f} h_out={d["h_out"]:.3f}')
    unit, E, eta, N, cos_a, tr_a = fast_evaluate_var(d['coords'], (d['xt'], d['yt']), d['W'], d['H'], d['h'])
    print(f'代理: unit={unit:.5f} E={E/1000:.3f}MW eta={eta:.5f} cos={cos_a:.5f} trunc={tr_a:.5f}')
    np.save(os.path.join(DATA_DIR, 'output', 'problem3_best.npy'), res.x)
    np.save(os.path.join(DATA_DIR, 'output', 'problem3_coords.npy'), d['coords'])
    np.save(os.path.join(DATA_DIR, 'output', 'problem3_sizes.npy'),
            np.column_stack([d['W'], d['H'], d['h']]))
    print('已保存')
