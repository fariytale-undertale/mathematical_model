# -*- coding: utf-8 -*-
"""标定阴影遮挡效率 sb 与布局间距/尺寸的关系"""
import os
import numpy as np
from helio import D_DAYS, TIMES, sun_direction_vector
from mirrors import (mirror_normal, mirror_basis, mirror_vertices, cosine_efficiency,
                     atmospheric_transmittance, reflected_direction, distance_to_tower,
                     TOWER_HEIGHT, RECV_DIAM, RECV_HEIGHT)
from raytrace import raytrace_time
from layout import generate_layout

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


def sb_annual(coords, W, H, h, tower_xy, K=60, seed=12345):
    """只算阴影遮挡效率的年均值 (全年60时刻)"""
    N = coords.shape[0]
    centers = np.column_stack([coords, np.full(N, h)])
    ncell, offsets, items = build_grid(centers)
    sb_list = []
    for mi in range(12):
        for ST in TIMES:
            s_dir = sun_direction_vector(D_DAYS[mi], ST)
            r = reflected_direction(centers, tower_xy)
            n = mirror_normal(s_dir[None, :], r)
            u, v = mirror_basis(n)
            verts = mirror_vertices(centers, n, W, H)
            rng = np.random.default_rng(seed)
            rnd = rng.random((N, K, 4)).astype(np.float64)
            shadow, block, hit = raytrace_time(
                verts, centers, u, v, n, s_dir.astype(np.float64), float(W), float(H),
                rnd, np.array(tower_xy), float(TOWER_HEIGHT), float(RECV_DIAM / 2.0),
                float(RECV_HEIGHT), ncell, offsets, items, BOUND, CELL_SIZE, MAX_CELLS)
            Kf = float(K)
            sb = (Kf - shadow - block) / Kf
            sb_list.append(sb.mean())
    return float(np.mean(sb_list))


if __name__ == '__main__':
    tower = (0.0, 0.0)
    for ds in [11, 13, 16, 20]:
        for W in [6.0]:
            dr = np.sqrt(3) / 2.0 * ds
            coords = generate_layout(0, 0, 100, dr, ds)
            sb = sb_annual(coords, W, W, 3.0, tower)
            print(f'ds={ds:.0f} W={W:.0f} N={coords.shape[0]} -> sb={sb:.4f}', flush=True)
