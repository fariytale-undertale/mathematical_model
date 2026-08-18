# -*- coding: utf-8 -*-
"""
问题1: 给定1745面镜子(6x6m, 高4m), 塔在圆心, 计算年平均光学效率与输出热功率
"""
import os
import time
import numpy as np
import openpyxl
from helio import D_DAYS, TIMES, sun_direction_vector, dni, RHO_REFLECT
from mirrors import (load_mirrors, mirror_normal, mirror_basis, mirror_vertices,
                     cosine_efficiency, atmospheric_transmittance, reflected_direction,
                     distance_to_tower, ATTACH_PATH, TOWER_HEIGHT, RECV_DIAM, RECV_HEIGHT)
from raytrace import raytrace_time

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELL_SIZE = 25.0
BOUND = 350.0
NCELL = int(np.ceil(2*BOUND/CELL_SIZE))   # 28
MAX_CELLS = 200


def build_grid(centers, cell_size=CELL_SIZE, bound=BOUND):
    ncell = int(np.ceil(2*bound/cell_size))
    ci = np.clip(((centers[:,0] + bound) / cell_size).astype(np.int64), 0, ncell-1)
    cj = np.clip(((centers[:,1] + bound) / cell_size).astype(np.int64), 0, ncell-1)
    cell_id = ci * ncell + cj
    order = np.argsort(cell_id, kind='stable')
    sorted_id = cell_id[order]
    counts = np.bincount(sorted_id, minlength=ncell*ncell)
    offsets = np.zeros(ncell*ncell+1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    items = order
    return ncell, offsets, items


def setup_problem1(width=6.0, height=6.0, install_h=4.0, tower_xy=(0.0, 0.0)):
    pts = load_mirrors(ATTACH_PATH)
    N = len(pts)
    centers = np.column_stack([pts, np.full(N, install_h)])
    ncell, offsets, items = build_grid(centers)
    return centers, ncell, offsets, items, N


def evaluate_time(centers, s_dir, tower_xy, width, height, K, seed,
                  ncell, offsets, items):
    """计算单时刻所有镜子的 shadow/block/hit 计数与解析效率"""
    N = centers.shape[0]
    r = reflected_direction(centers, tower_xy)
    n = mirror_normal(s_dir[None, :], r)
    u, v = mirror_basis(n)
    verts = mirror_vertices(centers, n, width, height)
    cos_eff = cosine_efficiency(s_dir[None, :], n)
    d_hr = distance_to_tower(centers, tower_xy)
    at_eff = atmospheric_transmittance(d_hr)

    rng = np.random.default_rng(seed)
    rnd = rng.random((N, K, 4)).astype(np.float64)

    shadow, block, hit = raytrace_time(
        verts, centers, u, v, n, s_dir.astype(np.float64), float(width), float(height),
        rnd, np.array(tower_xy, dtype=np.float64), float(TOWER_HEIGHT),
        float(RECV_DIAM/2.0), float(RECV_HEIGHT),
        ncell, offsets, items, BOUND, CELL_SIZE, MAX_CELLS)

    Kf = float(K)
    sb_eff = (Kf - shadow - block) / Kf
    trunc_eff = hit / np.maximum(Kf - shadow - block, 1e-9)
    # 被全部遮挡的镜子截断效率设为0
    trunc_eff = np.where(Kf - shadow - block < 1e-9, 0.0, trunc_eff)
    return cos_eff, at_eff, sb_eff, trunc_eff, d_hr


def run_problem1(K=100, seed_base=2023, verbose=True):
    """完整运行问题1: 60 时刻, 返回逐月/年平均结果"""
    centers, ncell, offsets, items, N = setup_problem1()
    tower_xy = (0.0, 0.0)
    width, height = 6.0, 6.0
    area_total = N * width * height

    # 逐月累计
    monthly = {}   # month_idx -> list of dict
    t_all = 0.0
    for mi in range(12):
        D = D_DAYS[mi]
        cos_list, sb_list, tr_list, eta_list, E_list, unit_list = [], [], [], [], [], []
        for ti, ST in enumerate(TIMES):
            s_dir = sun_direction_vector(D, ST)
            seed = seed_base + mi * 10 + ti
            t0 = time.time()
            cos_eff, at_eff, sb_eff, trunc_eff, d_hr = evaluate_time(
                centers, s_dir, tower_xy, width, height, K, seed, ncell, offsets, items)
            t_all += time.time() - t0
            eta = cos_eff * at_eff * sb_eff * trunc_eff * RHO_REFLECT
            DNI_t = dni(D, ST)
            E_kw = DNI_t * width * height * eta.sum()
            cos_list.append(cos_eff.mean())
            sb_list.append(sb_eff.mean())
            tr_list.append(trunc_eff.mean())
            eta_list.append(eta.mean())
            E_list.append(E_kw)
            unit_list.append(E_kw / area_total)
        monthly[mi] = dict(
            cos=float(np.mean(cos_list)), sb=float(np.mean(sb_list)),
            trunc=float(np.mean(tr_list)), eta=float(np.mean(eta_list)),
            E_kw=float(np.mean(E_list)), unit=float(np.mean(unit_list)))
        if verbose:
            print(f'  {mi+1:2d}月21日  eta={monthly[mi]["eta"]:.4f}  '
                  f'E={monthly[mi]["E_kw"]/1000:.3f} MW  unit={monthly[mi]["unit"]:.4f} kW/m2')
    # 年平均
    annual = dict(
        cos=float(np.mean([monthly[m]['cos'] for m in range(12)])),
        sb=float(np.mean([monthly[m]['sb'] for m in range(12)])),
        trunc=float(np.mean([monthly[m]['trunc'] for m in range(12)])),
        eta=float(np.mean([monthly[m]['eta'] for m in range(12)])),
        E_kw=float(np.mean([monthly[m]['E_kw'] for m in range(12)])),
        unit=float(np.mean([monthly[m]['unit'] for m in range(12)])),
    )
    return monthly, annual, t_all, N


if __name__ == '__main__':
    import json
    K = 100
    monthly, annual, t_all, N = run_problem1(K=K)
    print('\n===== 表2 年平均 (问题1) =====')
    print(f'  年平均光学效率      = {annual["eta"]:.4f}')
    print(f'  年平均余弦效率      = {annual["cos"]:.4f}')
    print(f'  年平均阴影遮挡效率  = {annual["sb"]:.4f}')
    print(f'  年平均截断效率      = {annual["trunc"]:.4f}')
    print(f'  年平均输出热功率    = {annual["E_kw"]/1000:.4f} MW')
    print(f'  单位面积年平均输出  = {annual["unit"]:.4f} kW/m2')
    print(f'  总用时 = {t_all:.1f}s, N = {N}')

    out = {'K': K, 'monthly': monthly, 'annual': annual, 'N': N}
    outpath = os.path.join(DATA_DIR, 'output', 'problem1_results.json')
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('结果已保存:', outpath)
