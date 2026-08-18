# -*- coding: utf-8 -*-
"""问题3 最终结果: 异尺寸完整验证 + 保存 + 写 result3.xlsx"""
import os
import json
import numpy as np
from helio import D_DAYS, TIMES, sun_direction_vector, dni, RHO_REFLECT
from mirrors import (mirror_normal, mirror_basis, cosine_efficiency,
                     atmospheric_transmittance, reflected_direction, distance_to_tower,
                     TOWER_HEIGHT, RECV_DIAM, RECV_HEIGHT)
from raytrace import raytrace_time_var
from layout import generate_layout
from write_results import write_result

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELL_SIZE = 25.0
BOUND = 350.0
MAX_CELLS = 200


def build_grid(centers):
    ncell = int(np.ceil(2 * BOUND / CELL_SIZE))
    ci = np.clip(((centers[:, 0] + BOUND) / CELL_SIZE).astype(np.int64), 0, ncell - 1)
    cj = np.clip(((centers[:, 1] + BOUND) / CELL_SIZE).astype(np.int64), 0, ncell - 1)
    cell_id = ci * ncell + cj
    order = np.argsort(cell_id, kind='stable')
    sorted_id = cell_id[order]
    counts = np.bincount(sorted_id, minlength=ncell * ncell)
    offsets = np.zeros(ncell * ncell + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    return ncell, offsets, order


def verify_layout_var(coords, W_arr, H_arr, h_arr, tower_xy, K=120, seed_base=2023):
    N = coords.shape[0]
    W_arr = np.asarray(W_arr, float); H_arr = np.asarray(H_arr, float); h_arr = np.asarray(h_arr, float)
    centers = np.column_stack([coords, h_arr])
    area = W_arr * H_arr
    ncell, offsets, items = build_grid(centers)
    monthly = []
    eta_all, E_all = [], []
    for mi in range(12):
        cos_l, sb_l, tr_l, eta_l, E_l = [], [], [], [], []
        for ti, ST in enumerate(TIMES):
            s_dir = sun_direction_vector(D_DAYS[mi], ST)
            r = reflected_direction(centers, tower_xy)
            n = mirror_normal(s_dir[None, :], r)
            u, v = mirror_basis(n)
            # 数组尺寸的顶点
            uu = u * (W_arr[:, None] / 2.0)
            vv = v * (H_arr[:, None] / 2.0)
            verts = np.stack([centers + uu + vv, centers - uu + vv,
                              centers - uu - vv, centers + uu - vv], axis=-2)
            cos_eff = cosine_efficiency(s_dir[None, :], n)
            d_hr = distance_to_tower(centers, tower_xy)
            at_eff = atmospheric_transmittance(d_hr)
            rng = np.random.default_rng(seed_base + mi * 10 + ti)
            rnd = rng.random((N, K, 4)).astype(np.float64)
            shadow, block, hit = raytrace_time_var(
                verts, centers, u, v, n, s_dir.astype(np.float64), W_arr, H_arr,
                rnd, np.array(tower_xy), float(TOWER_HEIGHT), float(RECV_DIAM / 2.0),
                float(RECV_HEIGHT), ncell, offsets, items, BOUND, CELL_SIZE, MAX_CELLS)
            Kf = float(K)
            sb_eff = (Kf - shadow - block) / Kf
            trunc_eff = hit / np.maximum(Kf - shadow - block, 1e-9)
            trunc_eff = np.where(Kf - shadow - block < 1e-9, 0.0, trunc_eff)
            eta = cos_eff * at_eff * sb_eff * trunc_eff * RHO_REFLECT
            DNI_t = dni(D_DAYS[mi], ST)
            E_kw_t = DNI_t * np.sum(area * eta)
            cos_l.append(cos_eff.mean()); sb_l.append(sb_eff.mean())
            tr_l.append(trunc_eff.mean()); eta_l.append(eta.mean()); E_l.append(E_kw_t)
        monthly.append(dict(cos=float(np.mean(cos_l)), sb=float(np.mean(sb_l)),
                            trunc=float(np.mean(tr_l)), eta=float(np.mean(eta_l)),
                            E_kw=float(np.mean(E_l)), unit=float(np.mean(E_l) / np.sum(area))))
        eta_all.extend(eta_l); E_all.extend(E_l)
    annual = dict(
        cos=float(np.mean([m['cos'] for m in monthly])),
        sb=float(np.mean([m['sb'] for m in monthly])),
        trunc=float(np.mean([m['trunc'] for m in monthly])),
        eta=float(np.mean(eta_all)),
        E_kw=float(np.mean(E_all)),
        unit=float(np.mean(E_all) / np.sum(area)),
    )
    return annual, monthly


if __name__ == '__main__':
    from problem3 import decode
    x = np.load(os.path.join(DATA_DIR, 'output', 'problem3_best.npy'))
    d = decode(x)
    tower = (d['xt'], d['yt'])
    coords = d['coords']; W = d['W']; H = d['H']; h = d['h']
    print(f'N={coords.shape[0]}')
    print(f'塔位={tower} W_in={d["W_in"]:.3f} W_out={d["W_out"]:.3f} '
          f'H_in={d["H_in"]:.3f} H_out={d["H_out"]:.3f} h_in={d["h_in"]:.3f} h_out={d["h_out"]:.3f}')
    annual, monthly = verify_layout_var(coords, W, H, h, tower, K=150)
    print(f'完整: eta={annual["eta"]:.5f} cos={annual["cos"]:.5f} sb={annual["sb"]:.5f} trunc={annual["trunc"]:.5f}')
    print(f'E={annual["E_kw"]/1000:.4f} MW  unit={annual["unit"]:.5f} kW/m2')
    result = dict(tower=tower, W_in=d['W_in'], W_out=d['W_out'], H_in=d['H_in'],
                  H_out=d['H_out'], h_in=d['h_in'], h_out=d['h_out'],
                  r0=d['r0'], ds=d['ds'], N=coords.shape[0], annual=annual, monthly=monthly)
    with open(os.path.join(DATA_DIR, 'output', 'problem3_result.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    write_result(os.path.join(DATA_DIR, 'result3.xlsx'), tower, coords, W, H, h)
    print('问题3 结果已保存')
