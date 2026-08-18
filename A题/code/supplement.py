# -*- coding: utf-8 -*-
"""补充计算: 基准模型 + W敏感性 + 塔位敏感性"""
import os
import json
import numpy as np
from layout import generate_layout
from verify import verify_layout

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL = [(mi, st) for mi in range(12) for st in [9.0, 10.5, 12.0, 13.5, 15.0]]


def benchmark():
    """问题二基准模型: 塔在原点, 6x6, 径向交错 ds=11.68"""
    tower = (0.0, 0.0)
    W, H, h = 6.0, 6.0, 4.0
    ds = 11.68
    dr = np.sqrt(3) / 2.0 * ds
    coords = generate_layout(0, 0, 100.0, dr, ds)
    annual, _ = verify_layout(coords, W, H, h, tower, K=120)
    return annual, coords.shape[0]


def sensitivity_W():
    """W 尺寸敏感性 (完整验证), 塔位固定问题2最优"""
    tower = (-4.99, -0.89)
    r0 = 101.73
    out = []
    for W in [6.0, 6.15, 6.30, 6.45, 6.60]:
        ds = W + 5.05
        dr = np.sqrt(3) / 2.0 * ds
        coords = generate_layout(tower[0], tower[1], r0, dr, ds)
        H = W * 0.995
        h = W / 2 + 0.15
        annual, _ = verify_layout(coords, W, H, h, tower, K=100)
        out.append(dict(W=W, N=coords.shape[0], E_kw=annual['E_kw'], unit=annual['unit'],
                        eta=annual['eta'], sb=annual['sb'], trunc=annual['trunc']))
        print(f'W={W:.2f} N={coords.shape[0]} E={annual["E_kw"]/1000:.3f}MW unit={annual["unit"]:.5f}', flush=True)
    return out


if __name__ == '__main__':
    print('=== 基准模型 ===', flush=True)
    a_b, n_b = benchmark()
    print(f'塔原点6x6: N={n_b} E={a_b["E_kw"]/1000:.3f}MW unit={a_b["unit"]:.5f} eta={a_b["eta"]:.5f}', flush=True)
    print('=== W 敏感性 ===', flush=True)
    sw = sensitivity_W()
    res = dict(benchmark=dict(N=n_b, **a_b), W_sensitivity=sw)
    with open(os.path.join(DATA_DIR, 'output', 'supplement.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print('已保存 supplement.json', flush=True)
