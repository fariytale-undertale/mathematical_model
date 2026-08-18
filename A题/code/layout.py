# -*- coding: utf-8 -*-
"""
径向交错布局生成 + 快速评估 (问题2/3优化用)
余弦/大气透射解析, 截断蒙特卡洛(trunc_eval), 阴影遮挡用常数
"""
import numpy as np
from helio import D_DAYS, TIMES, sun_direction_vector, dni, RHO_REFLECT
from mirrors import (mirror_normal, mirror_basis, cosine_efficiency,
                     atmospheric_transmittance, reflected_direction,
                     distance_to_tower, TOWER_HEIGHT, RECV_DIAM)
from raytrace import trunc_eval, trunc_eval_var

SB_CONST = 0.929        # 阴影遮挡常数 (问题1经验值, 优化代理)
RECV_R = RECV_DIAM / 2.0
K_TRUNC = 2


def sb_model(ds, W):
    """阴影遮挡效率经验模型 (径向交错布局标定): 损失 ≈ 2.02*(W/ds)^3.68"""
    ratio = W / ds
    loss = 2.02 * ratio ** 3.68
    return float(np.clip(1.0 - loss, 0.0, 1.0))


def generate_layout(xt, yt, r0, dr, ds, bound=350.0, r_min=100.0):
    """径向交错布局: 以塔(xt,yt)为圆心生成同心环, 截断到场地圆内"""
    coords = []
    k = 0
    r = r0
    while r <= bound:
        n_k = max(1, int(2 * np.pi * r / ds))
        offset = (k % 2) * np.pi / n_k
        for j in range(n_k):
            theta = 2 * np.pi * j / n_k + offset
            x = xt + r * np.cos(theta)
            y = yt + r * np.sin(theta)
            if x * x + y * y <= bound * bound:
                coords.append((x, y))
        r += dr
        k += 1
    return np.array(coords).reshape(-1, 2) if coords else np.zeros((0, 2))


def fast_evaluate(coords, tower_xy, W, H, h, K=K_TRUNC, seed=12345, times=None, sb=SB_CONST):
    """快速评估: 返回 (unit_power_kw_m2, E_kw, annual_eta, N, eta_cos_avg, eta_trunc_avg)
    times: 可选 [(mi, ST), ...] 默认用 12 个月正午 (快), 传入 None 用完整 60 时刻"""
    N = coords.shape[0]
    if N == 0:
        return 0.0, 0.0, 0.0, 0, 0.0, 0.0
    M = np.column_stack([coords, np.full(N, h)])
    area = W * H
    tower_xy = (float(tower_xy[0]), float(tower_xy[1]))
    rng = np.random.default_rng(seed)
    if times is None:
        times = [(mi, 12.0) for mi in range(12)]   # 12 个月正午

    eta_times, E_times, cos_avg, tr_avg = [], [], [], []
    for (mi, ST) in times:
        s = sun_direction_vector(D_DAYS[mi], ST)
        s_rep = np.tile(s, (N, 1))
        r = reflected_direction(M, tower_xy)
        n = mirror_normal(s_rep, r)
        u, v = mirror_basis(n)
        cos_eff = cosine_efficiency(s_rep, n)
        d_hr = distance_to_tower(M, tower_xy)
        at_eff = atmospheric_transmittance(d_hr)
        # 截断蒙特卡洛
        rnd = rng.random((N, K, 4)).astype(np.float64)
        hit = trunc_eval(M, u, v, n, s.astype(np.float64), float(W), float(H),
                         rnd, np.array(tower_xy), float(TOWER_HEIGHT),
                         float(RECV_R), float(8.0))
        trunc_eff = hit.astype(np.float64) / K
        eta = cos_eff * at_eff * trunc_eff * sb * RHO_REFLECT
        DNI_t = dni(D_DAYS[mi], ST)
        eta_times.append(eta.mean())
        E_times.append(DNI_t * area * eta.sum())
        cos_avg.append(cos_eff.mean())
        tr_avg.append(trunc_eff.mean())
    annual_eta = float(np.mean(eta_times))
    E_kw = float(np.mean(E_times))
    unit_power = E_kw / (N * area)
    return unit_power, E_kw, annual_eta, N, float(np.mean(cos_avg)), float(np.mean(tr_avg))


def fast_evaluate_var(coords, tower_xy, W_arr, H_arr, h_arr, K=K_TRUNC, seed=12345, sb=SB_CONST):
    """异尺寸快速评估: W_arr,H_arr,h_arr 为 (N,) 数组
    返回 (unit_power, E_kw, annual_eta, N, cos_avg, trunc_avg)"""
    N = coords.shape[0]
    if N == 0:
        return 0.0, 0.0, 0.0, 0, 0.0, 0.0
    W_arr = np.asarray(W_arr, dtype=np.float64)
    H_arr = np.asarray(H_arr, dtype=np.float64)
    h_arr = np.asarray(h_arr, dtype=np.float64)
    M = np.column_stack([coords, h_arr])
    area = W_arr * H_arr
    tower_xy = (float(tower_xy[0]), float(tower_xy[1]))
    rng = np.random.default_rng(seed)

    eta_times, E_times, cos_avg, tr_avg = [], [], [], []
    for mi in range(12):
        for ST in [9.0, 10.5, 12.0, 13.5, 15.0]:
            s = sun_direction_vector(D_DAYS[mi], ST)
            s_rep = np.tile(s, (N, 1))
            r = reflected_direction(M, tower_xy)
            n = mirror_normal(s_rep, r)
            u, v = mirror_basis(n)
            cos_eff = cosine_efficiency(s_rep, n)
            d_hr = distance_to_tower(M, tower_xy)
            at_eff = atmospheric_transmittance(d_hr)
            rnd = rng.random((N, K, 4)).astype(np.float64)
            hit = trunc_eval_var(M, u, v, n, s.astype(np.float64), W_arr, H_arr,
                                 rnd, np.array(tower_xy), float(TOWER_HEIGHT),
                                 float(RECV_R), float(8.0))
            trunc_eff = hit.astype(np.float64) / K
            eta = cos_eff * at_eff * trunc_eff * sb * RHO_REFLECT
            DNI_t = dni(D_DAYS[mi], ST)
            eta_times.append(eta.mean())
            E_times.append(DNI_t * np.sum(area * eta))
            cos_avg.append(cos_eff.mean())
            tr_avg.append(trunc_eff.mean())
    annual_eta = float(np.mean(eta_times))
    E_kw = float(np.mean(E_times))
    unit_power = E_kw / float(np.sum(area))
    return unit_power, E_kw, annual_eta, N, float(np.mean(cos_avg)), float(np.mean(tr_avg))


def generate_layout_ring(xt, yt, r0, W_in, W_out, bound=350.0, margin=5.05):
    """每环独立尺寸与间距的径向交错布局: 尺寸随半径线性变化
    返回 (coords(N,2), W_arr(N,))"""
    coords = []
    Ws = []
    k = 0
    r = r0
    while r <= bound:
        W_k = W_in + (W_out - W_in) * (r - r0) / max(bound - r0, 1e-9)
        ds_k = W_k + margin
        n_k = max(1, int(2 * np.pi * r / ds_k))
        offset = (k % 2) * np.pi / n_k
        for j in range(n_k):
            theta = 2 * np.pi * j / n_k + offset
            x = xt + r * np.cos(theta)
            y = yt + r * np.sin(theta)
            if x * x + y * y <= bound * bound:
                coords.append((x, y))
                Ws.append(W_k)
        r += np.sqrt(3) / 2.0 * ds_k
        k += 1
    if not coords:
        return np.zeros((0, 2)), np.zeros(0)
    return np.array(coords), np.array(Ws)


def generate_layout_origin(r0, dr, ds, tower_xy, bound=350.0, r_min=100.0):
    """以原点为中心的径向交错布局, 移除吸收塔周围 r_min 内的镜子
    (正确建模: 场地以原点为中心, 塔可偏移)"""
    coords = []
    r = r0
    k = 0
    while r <= bound:
        n_k = max(1, int(2 * np.pi * r / ds))
        offset = (k % 2) * np.pi / n_k
        for j in range(n_k):
            theta = 2 * np.pi * j / n_k + offset
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            if (x - tower_xy[0]) ** 2 + (y - tower_xy[1]) ** 2 >= r_min ** 2:
                coords.append((x, y))
        r += dr
        k += 1
    return np.array(coords).reshape(-1, 2) if coords else np.zeros((0, 2))


def generate_layout_origin_ring(r0, W_in, W_out, tower_xy, bound=350.0, margin=5.05, r_min=100.0):
    """以原点为中心, 尺寸随半径线性变化, 每环独立间距的径向交错布局
    返回 (coords(N,2), W_arr(N,))"""
    coords = []
    Ws = []
    k = 0
    r = r0
    while r <= bound:
        W_k = W_in + (W_out - W_in) * (r - r0) / max(bound - r0, 1e-9)
        ds_k = W_k + margin
        n_k = max(1, int(2 * np.pi * r / ds_k))
        offset = (k % 2) * np.pi / n_k
        for j in range(n_k):
            theta = 2 * np.pi * j / n_k + offset
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            if (x - tower_xy[0]) ** 2 + (y - tower_xy[1]) ** 2 >= r_min ** 2:
                coords.append((x, y))
                Ws.append(W_k)
        r += np.sqrt(3) / 2.0 * ds_k
        k += 1
    if not coords:
        return np.zeros((0, 2)), np.zeros(0)
    return np.array(coords), np.array(Ws)
