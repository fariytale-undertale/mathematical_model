# -*- coding: utf-8 -*-
"""完整光线追踪验证 (含真实阴影遮挡)"""
import os
import time
import numpy as np
from helio import D_DAYS, TIMES, sun_direction_vector, dni, RHO_REFLECT
from mirrors import (mirror_normal, mirror_basis, mirror_vertices, cosine_efficiency,
                     atmospheric_transmittance, reflected_direction, distance_to_tower,
                     TOWER_HEIGHT, RECV_DIAM, RECV_HEIGHT)
from raytrace import raytrace_time

CELL_SIZE = 25.0
BOUND = 350.0
MAX_CELLS = 200


def build_grid(centers, cell_size=CELL_SIZE, bound=BOUND):
    ncell = int(np.ceil(2 * bound / cell_size))
    ci = np.clip(((centers[:, 0] + bound) / cell_size).astype(np.int64), 0, ncell - 1)
    cj = np.clip(((centers[:, 1] + bound) / cell_size).astype(np.int64), 0, ncell - 1)
    cell_id = ci * ncell + cj
    order = np.argsort(cell_id, kind='stable')
    sorted_id = cell_id[order]
    counts = np.bincount(sorted_id, minlength=ncell * ncell)
    offsets = np.zeros(ncell * ncell + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    return ncell, offsets, order


def verify_layout(coords, W, H, h, tower_xy, K=100, seed_base=2023):
    """完整光线追踪验证同尺寸布局, 返回 (unit, E_kw, annual_eta, monthly, annual_detail)"""
    N = coords.shape[0]
    centers = np.column_stack([coords, np.full(N, h)])
    area = W * H
    ncell, offsets, items = build_grid(centers)
    tower_xy = (float(tower_xy[0]), float(tower_xy[1]))

    monthly = []
    eta_all, E_all = [], []
    for mi in range(12):
        cos_l, sb_l, tr_l, eta_l, E_l = [], [], [], [], []
        for ti, ST in enumerate(TIMES):
            s_dir = sun_direction_vector(D_DAYS[mi], ST)
            r = reflected_direction(centers, tower_xy)
            n = mirror_normal(s_dir[None, :], r)
            u, v = mirror_basis(n)
            verts = mirror_vertices(centers, n, W, H)
            cos_eff = cosine_efficiency(s_dir[None, :], n)
            d_hr = distance_to_tower(centers, tower_xy)
            at_eff = atmospheric_transmittance(d_hr)
            rng = np.random.default_rng(seed_base + mi * 10 + ti)
            rnd = rng.random((N, K, 4)).astype(np.float64)
            shadow, block, hit = raytrace_time(
                verts, centers, u, v, n, s_dir.astype(np.float64), float(W), float(H),
                rnd, np.array(tower_xy), float(TOWER_HEIGHT), float(RECV_DIAM / 2.0),
                float(RECV_HEIGHT), ncell, offsets, items, BOUND, CELL_SIZE, MAX_CELLS)
            Kf = float(K)
            sb_eff = (Kf - shadow - block) / Kf
            trunc_eff = hit / np.maximum(Kf - shadow - block, 1e-9)
            trunc_eff = np.where(Kf - shadow - block < 1e-9, 0.0, trunc_eff)
            eta = cos_eff * at_eff * sb_eff * trunc_eff * RHO_REFLECT
            DNI_t = dni(D_DAYS[mi], ST)
            E_kw_t = DNI_t * area * eta.sum()
            cos_l.append(cos_eff.mean()); sb_l.append(sb_eff.mean())
            tr_l.append(trunc_eff.mean()); eta_l.append(eta.mean()); E_l.append(E_kw_t)
        monthly.append(dict(cos=float(np.mean(cos_l)), sb=float(np.mean(sb_l)),
                            trunc=float(np.mean(tr_l)), eta=float(np.mean(eta_l)),
                            E_kw=float(np.mean(E_l)), unit=float(np.mean(E_l) / (N * area))))
        eta_all.extend(eta_l); E_all.extend(E_l)
    annual = dict(
        cos=float(np.mean([m['cos'] for m in monthly])),
        sb=float(np.mean([m['sb'] for m in monthly])),
        trunc=float(np.mean([m['trunc'] for m in monthly])),
        eta=float(np.mean(eta_all)),
        E_kw=float(np.mean(E_all)),
        unit=float(np.mean(E_all) / (N * area)),
    )
    return annual, monthly


if __name__ == '__main__':
    import json
    DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    which = 2
    if which == 2:
        from problem2 import decode
        x = np.load(os.path.join(DATA_DIR, 'output', 'problem2_best.npy'))
        d = decode(x)
        coords, W, H, h = d['coords'], d['W'], d['H'], d['h']
        tower = (d['xt'], d['yt'])
    else:
        from problem3 import decode
        x = np.load(os.path.join(DATA_DIR, 'output', 'problem3_best.npy'))
        d = decode(x)
        coords = d['coords']
        # 同尺寸近似用均值验证 (异尺寸完整验证单独处理)
        W, H, h = d['W'].mean(), d['H'].mean(), d['h'].mean()
        tower = (d['xt'], d['yt'])
    t0 = time.time()
    annual, monthly = verify_layout(coords, W, H, h, tower, K=100)
    print(f'完整验证用时 {time.time()-t0:.1f}s, N={coords.shape[0]}')
    print(f'塔位={tower}  W={W:.3f} H={H:.3f} h={h:.3f}')
    print(f'年平均: eta={annual["eta"]:.5f} cos={annual["cos"]:.5f} sb={annual["sb"]:.5f} trunc={annual["trunc"]:.5f}')
    print(f'E={annual["E_kw"]/1000:.4f} MW  unit={annual["unit"]:.5f} kW/m2')
