# -*- coding: utf-8 -*-
"""
镜面几何与解析效率(余弦/大气透射)模块
坐标: x 正东, y 正北, z 向上
"""
import os
import numpy as np
import openpyxl
from helio import sun_direction_vector, dni

# 数据目录 = 脚本所在目录的上级 (A题目录)
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATTACH_PATH = os.path.join(DATA_DIR, '附件.xlsx')

# 集热器 (吸收塔) 参数
TOWER_HEIGHT = 80.0       # 集热器中心离地高度 (m)
RECV_DIAM = 7.0           # 集热器直径 (m)
RECV_HEIGHT = 8.0         # 集热器高度 (m)
R_MIN = 100.0             # 厂房内圈半径 (m)
R_MAX = 350.0             # 镜场半径 (m)
MIN_SPACING_EXTRA = 5.0   # 相邻底座中心距 > 宽度 + 5


def load_mirrors(path):
    """读取附件定日镜位置 (x, y)"""
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb['Sheet1']
    pts = [(r[0], r[1]) for r in ws.iter_rows(min_row=2, values_only=True)]
    return np.array(pts, dtype=float)   # (N,2)


def mirror_normal(s, r):
    """镜面法向 = 归一化(入射方向 + 反射方向)"""
    n = s + r
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def mirror_basis(n):
    """
    返回镜面局部基 (u, v): u 水平(宽度方向), v 倾斜(高度方向), 使上下边平行地面.
    n: (...,3) 法向单位向量
    """
    # u = normalize(z_hat x n) = normalize(-ny, nx, 0)
    n_h_norm = np.linalg.norm(n[..., :2], axis=-1, keepdims=True)
    n_h_norm = np.maximum(n_h_norm, 1e-12)
    u = np.stack([-n[..., 1], n[..., 0], np.zeros_like(n[..., 0])], axis=-1) / n_h_norm
    # v = n x u
    v = np.cross(n, u)
    return u, v


def mirror_vertices(M, n, width, height):
    """
    镜面四个顶点. M:(...,3) 中心, n:(...,3) 法向, width:宽度(水平), height:高度
    返回 (...,4,3)
    """
    u, v = mirror_basis(n)
    u = u * (width / 2.0)
    v = v * (height / 2.0)
    M = np.asarray(M)
    # 四个角: M + a*u + b*v, a,b in {+1,-1}
    verts = np.stack([
        M + u + v,
        M - u + v,
        M - u - v,
        M + u - v,
    ], axis=-2)
    return verts


def cosine_efficiency(s, n):
    """余弦效率 = 入射光与法向夹角余弦"""
    return np.clip(np.sum(s * n, axis=-1), 0.0, 1.0)


def atmospheric_transmittance(d_hr):
    """大气透射率 (d_hr: 镜面中心到集热器中心距离 m)"""
    return 0.99321 - 0.0001176 * d_hr + 1.97e-8 * d_hr ** 2


def reflected_direction(M, tower_xy, tower_height=TOWER_HEIGHT):
    """反射方向 r: 从镜面中心指向集热器中心"""
    T = np.array([tower_xy[0], tower_xy[1], tower_height])
    d = T - M
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def distance_to_tower(M, tower_xy, tower_height=TOWER_HEIGHT):
    """镜面中心到集热器中心距离"""
    T = np.array([tower_xy[0], tower_xy[1], tower_height])
    return np.linalg.norm(T - M, axis=-1)


if __name__ == '__main__':
    # 量级 sanity check: 用附件数据算余弦效率与大气透射率 (正午)
    from helio import D_DAYS, TIMES
    pts = load_mirrors(ATTACH_PATH)
    N = len(pts)
    z = 4.0
    M = np.column_stack([pts, np.full(N, z)])
    tower_xy = (0.0, 0.0)

    s = sun_direction_vector(D_DAYS[5], 12.0)   # 6月21日 正午
    s = np.tile(s, (N, 1))
    r = reflected_direction(M, tower_xy)
    n = mirror_normal(s, r)
    cos_eff = cosine_efficiency(s, n)
    d_hr = distance_to_tower(M, tower_xy)
    at_eff = atmospheric_transmittance(d_hr)

    print(f'N = {N}')
    print(f'余弦效率: min={cos_eff.min():.4f} max={cos_eff.max():.4f} mean={cos_eff.mean():.4f}')
    print(f'大气透射率: min={at_eff.min():.4f} max={at_eff.max():.4f} mean={at_eff.mean():.4f}')
    print(f'距离d_HR: min={d_hr.min():.1f} max={d_hr.max():.1f} m')
    # 简单估算 (忽略阴影遮挡与截断, 只看 eta_cos*eta_at*0.92)
    eta_approx = cos_eff * at_eff * 0.92
    area = 36.0 * N
    import helio
    E = helio.dni(D_DAYS[5], 12.0) * area * eta_approx.mean()
    print(f'粗略 E_field (6月正午, 忽略sb/trunc) = {E/1000:.1f} MW')
