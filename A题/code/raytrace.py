# -*- coding: utf-8 -*-
"""
蒙特卡洛光线追踪模块 (numba 加速)
计算阴影(shadowing)/遮挡(blocking)效率与截断效率
统一一次采样: 阴影->遮挡->截断
"""
import numpy as np
from numba import njit, prange

SUN_HALF_ANGLE = 4.65e-3   # 太阳锥半张角 (rad)
EPS = 1e-6
INF = 1e30


@njit(cache=False, inline='always')
def ray_triangle(ox, oy, oz, dx, dy, dz, v0, v1, v2):
    """Möller–Trumbore 光线-三角形求交, 返回 t (>=EPS) 或 INF"""
    e1x, e1y, e1z = v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]
    e2x, e2y, e2z = v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]
    # pvec = d x e2
    px = dy*e2z - dz*e2y
    py = dz*e2x - dx*e2z
    pz = dx*e2y - dy*e2x
    det = e1x*px + e1y*py + e1z*pz
    if det > -1e-12 and det < 1e-12:
        return INF
    inv_det = 1.0/det
    # tvec = o - v0
    tx = ox - v0[0]; ty = oy - v0[1]; tz = oz - v0[2]
    u = (tx*px + ty*py + tz*pz) * inv_det
    if u < 0.0 or u > 1.0:
        return INF
    # qvec = tvec x e1
    qx = ty*e1z - tz*e1y
    qy = tz*e1x - tx*e1z
    qz = tx*e1y - ty*e1x
    v = (dx*qx + dy*qy + dz*qz) * inv_det
    if v < 0.0 or u + v > 1.0:
        return INF
    t = (e2x*qx + e2y*qy + e2z*qz) * inv_det
    if t > EPS:
        return t
    return INF


@njit(cache=False, inline='always')
def ray_rect(ox, oy, oz, dx, dy, dz, verts):
    """光线-矩形求交 (两个三角形)"""
    t1 = ray_triangle(ox, oy, oz, dx, dy, dz, verts[0], verts[1], verts[2])
    t2 = ray_triangle(ox, oy, oz, dx, dy, dz, verts[0], verts[2], verts[3])
    return t1 if t1 < t2 else t2


@njit(cache=False, inline='always')
def ray_cylinder(ox, oy, oz, dx, dy, dz, cx, cy, cz, R, H):
    """光线-竖直圆柱求交 (侧面+上下底), 返回最近 t (>=EPS) 或 INF
    圆柱轴 (cx,cy), z范围 [cz-H/2, cz+H/2], 半径 R"""
    zlo = cz - H/2.0
    zhi = cz + H/2.0
    best = INF
    # 侧面
    A = dx*dx + dy*dy
    if A > 1e-18:
        px = ox - cx; py = oy - cy
        B = 2.0*(px*dx + py*dy)
        C = px*px + py*py - R*R
        disc = B*B - 4.0*A*C
        if disc >= 0.0:
            sq = np.sqrt(disc)
            for t in (( -B - sq)/(2.0*A), (-B + sq)/(2.0*A)):
                if t > EPS and t < best:
                    z = oz + t*dz
                    if zlo <= z <= zhi:
                        best = t
    # 上底 z=zhi (圆盘)
    if dz > 1e-18:
        t = (zhi - oz)/dz
        if t > EPS and t < best:
            x = ox + t*dx; y = oy + t*dy
            if (x-cx)**2 + (y-cy)**2 <= R*R:
                best = t
    # 下底 z=zlo
    if dz < -1e-18:
        t = (zlo - oz)/dz
        if t > EPS and t < best:
            x = ox + t*dx; y = oy + t*dy
            if (x-cx)**2 + (y-cy)**2 <= R*R:
                best = t
    return best


@njit(cache=False, inline='always')
def dda_traverse(ox, oy, dx, dy, t_max, bound, cell_size, ncell, out, max_cells):
    """2D DDA 遍历光线在 xy 投影经过的网格单元, 返回单元数
    out: 预分配数组, 存放 cell 线性索引 (i*ncell+j)"""
    n = 0
    if dx == 0.0 and dy == 0.0:
        return 0
    minc = -bound
    # 起点 cell
    ix = int((ox - minc) / cell_size)
    iy = int((oy - minc) / cell_size)
    if ix < 0 or ix >= ncell or iy < 0 or iy >= ncell:
        return 0
    stepx = 1 if dx > 0 else (-1 if dx < 0 else 0)
    stepy = 1 if dy > 0 else (-1 if dy < 0 else 0)
    # tMax: 到下一 cell 边界的时间
    if dx != 0.0:
        next_x = minc + (ix + (1 if dx > 0 else 0)) * cell_size
        tMaxX = (next_x - ox) / dx
        tDeltaX = cell_size / abs(dx)
    else:
        tMaxX = INF; tDeltaX = INF
    if dy != 0.0:
        next_y = minc + (iy + (1 if dy > 0 else 0)) * cell_size
        tMaxY = (next_y - oy) / dy
        tDeltaY = cell_size / abs(dy)
    else:
        tMaxY = INF; tDeltaY = INF

    t = 0.0
    while True:
        if n >= max_cells or t > t_max:
            break
        if ix < 0 or ix >= ncell or iy < 0 or iy >= ncell:
            break
        out[n] = ix * ncell + iy
        n += 1
        if tMaxX < tMaxY:
            t = tMaxX; tMaxX += tDeltaX; ix += stepx
        else:
            t = tMaxY; tMaxY += tDeltaY; iy += stepy
    return n


@njit(cache=False, inline='always')
def hit_mirror(ox, oy, oz, dx, dy, dz, self_idx, verts, ncell, offsets, items,
               bound, cell_size, out, max_cells):
    """从 (ox,oy,oz) 沿 d 发射, 检查是否撞到其他镜子 (排除 self_idx), 返回是否撞到"""
    dxh = np.sqrt(dx*dx + dy*dy)
    if dxh < 1e-12:
        # 纯竖直光线, 不会撞到水平镜面(镜面近似水平/倾斜), 保守返回 False
        return False
    # 水平投影 t_max: 到场地边界
    n = dda_traverse(ox, oy, dx, dy, 2.0*bound, bound, cell_size, ncell, out, max_cells)
    for k in range(n):
        c = out[k]
        for idx in range(offsets[c], offsets[c+1]):
            j = items[idx]
            if j == self_idx:
                continue
            t = ray_rect(ox, oy, oz, dx, dy, dz, verts[j])
            if t < INF:
                return True
    return False


@njit(cache=False, parallel=True)
def raytrace_time(verts, centers, u_vec, v_vec, n_norm, s_dir, width, height,
                  rnd, tower_xy, tower_h, recv_R, recv_H,
                  ncell, offsets, items, bound, cell_size, max_cells):
    """
    对单个时刻做蒙特卡洛光线追踪
    verts: (N,4,3) 顶点; centers:(N,3); u_vec,v_vec:(N,3) 镜面基
    n_norm:(N,3) 法向; s_dir:(3,) 中心太阳方向(镜面->太阳)
    width,height: 标量或数组; rnd:(N,K,4) 随机数
    返回 shadow_count, block_count, hit_count (N,) int
    """
    N = verts.shape[0]
    K = rnd.shape[1]
    shadow = np.zeros(N, dtype=np.int64)
    block = np.zeros(N, dtype=np.int64)
    hit = np.zeros(N, dtype=np.int64)

    cx = tower_xy[0]; cy = tower_xy[1]; cz = tower_h
    tan_eps = np.tan(SUN_HALF_ANGLE)

    # 中心太阳方向的垂直基 (用于锥采样)
    zhat = np.array([0.0, 0.0, 1.0])
    # e1 = normalize(s x zhat)
    e1 = np.array([s_dir[1]*zhat[2]-s_dir[2]*zhat[1],
                   s_dir[2]*zhat[0]-s_dir[0]*zhat[2],
                   s_dir[0]*zhat[1]-s_dir[1]*zhat[0]])
    e1n = np.sqrt(e1[0]**2+e1[1]**2+e1[2]**2)
    if e1n < 1e-12:
        e1 = np.array([1.0, 0.0, 0.0])
    else:
        e1 = e1 / e1n
    e2 = np.array([s_dir[1]*e1[2]-s_dir[2]*e1[1],
                   s_dir[2]*e1[0]-s_dir[0]*e1[2],
                   s_dir[0]*e1[1]-s_dir[1]*e1[0]])

    w = width
    h = height

    for i in prange(N):
        out = np.empty(max_cells, dtype=np.int64)   # 每线程独立缓冲区, 避免数据竞争
        cx0, cy0, cz0 = centers[i,0], centers[i,1], centers[i,2]
        nix, niy, niz = n_norm[i,0], n_norm[i,1], n_norm[i,2]
        uix, uiy, uiz = u_vec[i,0], u_vec[i,1], u_vec[i,2]
        vix, viy, viz = v_vec[i,0], v_vec[i,1], v_vec[i,2]
        for k in range(K):
            a = rnd[i,k,0] * 2.0 - 1.0   # [-1,1]
            b = rnd[i,k,1] * 2.0 - 1.0
            # 镜面采样点
            px = cx0 + (w/2.0)*a*uix + (h/2.0)*b*vix
            py = cy0 + (w/2.0)*a*uiy + (h/2.0)*b*viy
            pz = cz0 + (w/2.0)*a*uiz + (h/2.0)*b*viz
            # 锥内采样入射方向 s'
            rho = np.sqrt(rnd[i,k,2]) * tan_eps
            phi = rnd[i,k,3] * 2.0 * np.pi
            sx = s_dir[0] + rho*(np.cos(phi)*e1[0] + np.sin(phi)*e2[0])
            sy = s_dir[1] + rho*(np.cos(phi)*e1[1] + np.sin(phi)*e2[1])
            sz = s_dir[2] + rho*(np.cos(phi)*e1[2] + np.sin(phi)*e2[2])
            sn = np.sqrt(sx*sx+sy*sy+sz*sz)
            sx /= sn; sy /= sn; sz /= sn
            # 阴影: 从 p 沿 +s' 发射 (朝太阳, 检查入射光路是否被其他镜子挡住)
            if hit_mirror(px, py, pz, sx, sy, sz, i, verts, ncell, offsets, items,
                          bound, cell_size, out, max_cells):
                shadow[i] += 1
                continue
            # 反射方向 r' = -s' + 2(s'·n)n  (d_out = d_in - 2(d_in·n)n, d_in = -s')
            sd = sx*nix + sy*niy + sz*niz
            rx = -sx + 2.0*sd*nix
            ry = -sy + 2.0*sd*niy
            rz = -sz + 2.0*sd*niz
            # 遮挡: 从 p 沿 r' 发射
            if hit_mirror(px, py, pz, rx, ry, rz, i, verts, ncell, offsets, items,
                          bound, cell_size, out, max_cells):
                block[i] += 1
                continue
            # 截断: 命中集热器?
            t = ray_cylinder(px, py, pz, rx, ry, rz, cx, cy, cz, recv_R, recv_H)
            if t < INF:
                hit[i] += 1
    return shadow, block, hit


@njit(cache=False, parallel=True)
def trunc_eval(centers, u_vec, v_vec, n_norm, s_dir, width, height, rnd,
               tower_xy, tower_h, recv_R, recv_H):
    """只计算截断效率: 对每面镜子锥采样, 反射后检查是否命中集热器
    rnd:(N,K,4) 随机数; 返回 (N,) hit 计数"""
    N = centers.shape[0]
    K = rnd.shape[1]
    hit = np.zeros(N, dtype=np.int64)
    cx = tower_xy[0]; cy = tower_xy[1]; cz = tower_h
    tan_eps = np.tan(SUN_HALF_ANGLE)
    zhat = np.array([0.0, 0.0, 1.0])
    e1 = np.array([s_dir[1]*zhat[2]-s_dir[2]*zhat[1],
                   s_dir[2]*zhat[0]-s_dir[0]*zhat[2],
                   s_dir[0]*zhat[1]-s_dir[1]*zhat[0]])
    e1n = np.sqrt(e1[0]**2+e1[1]**2+e1[2]**2)
    if e1n < 1e-12:
        e1 = np.array([1.0, 0.0, 0.0])
    else:
        e1 = e1 / e1n
    e2 = np.array([s_dir[1]*e1[2]-s_dir[2]*e1[1],
                   s_dir[2]*e1[0]-s_dir[0]*e1[2],
                   s_dir[0]*e1[1]-s_dir[1]*e1[0]])
    w = width; h = height
    for i in prange(N):
        cx0, cy0, cz0 = centers[i,0], centers[i,1], centers[i,2]
        nix, niy, niz = n_norm[i,0], n_norm[i,1], n_norm[i,2]
        uix, uiy, uiz = u_vec[i,0], u_vec[i,1], u_vec[i,2]
        vix, viy, viz = v_vec[i,0], v_vec[i,1], v_vec[i,2]
        for k in range(K):
            a = rnd[i,k,0]*2.0-1.0
            b = rnd[i,k,1]*2.0-1.0
            px = cx0 + (w/2.0)*a*uix + (h/2.0)*b*vix
            py = cy0 + (w/2.0)*a*uiy + (h/2.0)*b*viy
            pz = cz0 + (w/2.0)*a*uiz + (h/2.0)*b*viz
            rho = np.sqrt(rnd[i,k,2]) * tan_eps
            phi = rnd[i,k,3] * 2.0*np.pi
            sx = s_dir[0] + rho*(np.cos(phi)*e1[0] + np.sin(phi)*e2[0])
            sy = s_dir[1] + rho*(np.cos(phi)*e1[1] + np.sin(phi)*e2[1])
            sz = s_dir[2] + rho*(np.cos(phi)*e1[2] + np.sin(phi)*e2[2])
            sn = np.sqrt(sx*sx+sy*sy+sz*sz)
            sx /= sn; sy /= sn; sz /= sn
            sd = sx*nix + sy*niy + sz*niz
            rx = -sx + 2.0*sd*nix
            ry = -sy + 2.0*sd*niy
            rz = -sz + 2.0*sd*niz
            if ray_cylinder(px, py, pz, rx, ry, rz, cx, cy, cz, recv_R, recv_H) < INF:
                hit[i] += 1
    return hit


@njit(cache=False, parallel=True)
def trunc_eval_var(centers, u_vec, v_vec, n_norm, s_dir, width, height, rnd,
                   tower_xy, tower_h, recv_R, recv_H):
    """支持逐镜尺寸数组的截断评估. width,height: (N,) 数组
    返回 (N,) hit 计数"""
    N = centers.shape[0]
    K = rnd.shape[1]
    hit = np.zeros(N, dtype=np.int64)
    cx = tower_xy[0]; cy = tower_xy[1]; cz = tower_h
    tan_eps = np.tan(SUN_HALF_ANGLE)
    zhat = np.array([0.0, 0.0, 1.0])
    e1 = np.array([s_dir[1]*zhat[2]-s_dir[2]*zhat[1],
                   s_dir[2]*zhat[0]-s_dir[0]*zhat[2],
                   s_dir[0]*zhat[1]-s_dir[1]*zhat[0]])
    e1n = np.sqrt(e1[0]**2+e1[1]**2+e1[2]**2)
    if e1n < 1e-12:
        e1 = np.array([1.0, 0.0, 0.0])
    else:
        e1 = e1 / e1n
    e2 = np.array([s_dir[1]*e1[2]-s_dir[2]*e1[1],
                   s_dir[2]*e1[0]-s_dir[0]*e1[2],
                   s_dir[0]*e1[1]-s_dir[1]*e1[0]])
    for i in prange(N):
        w = width[i]; h = height[i]
        cx0, cy0, cz0 = centers[i,0], centers[i,1], centers[i,2]
        nix, niy, niz = n_norm[i,0], n_norm[i,1], n_norm[i,2]
        uix, uiy, uiz = u_vec[i,0], u_vec[i,1], u_vec[i,2]
        vix, viy, viz = v_vec[i,0], v_vec[i,1], v_vec[i,2]
        for k in range(K):
            a = rnd[i,k,0]*2.0-1.0
            b = rnd[i,k,1]*2.0-1.0
            px = cx0 + (w/2.0)*a*uix + (h/2.0)*b*vix
            py = cy0 + (w/2.0)*a*uiy + (h/2.0)*b*viy
            pz = cz0 + (w/2.0)*a*uiz + (h/2.0)*b*viz
            rho = np.sqrt(rnd[i,k,2]) * tan_eps
            phi = rnd[i,k,3] * 2.0*np.pi
            sx = s_dir[0] + rho*(np.cos(phi)*e1[0] + np.sin(phi)*e2[0])
            sy = s_dir[1] + rho*(np.cos(phi)*e1[1] + np.sin(phi)*e2[1])
            sz = s_dir[2] + rho*(np.cos(phi)*e1[2] + np.sin(phi)*e2[2])
            sn = np.sqrt(sx*sx+sy*sy+sz*sz)
            sx /= sn; sy /= sn; sz /= sn
            sd = sx*nix + sy*niy + sz*niz
            rx = -sx + 2.0*sd*nix
            ry = -sy + 2.0*sd*niy
            rz = -sz + 2.0*sd*niz
            if ray_cylinder(px, py, pz, rx, ry, rz, cx, cy, cz, recv_R, recv_H) < INF:
                hit[i] += 1
    return hit


@njit(cache=False, parallel=True)
def raytrace_time_var(verts, centers, u_vec, v_vec, n_norm, s_dir, width, height,
                      rnd, tower_xy, tower_h, recv_R, recv_H,
                      ncell, offsets, items, bound, cell_size, max_cells):
    """支持逐镜尺寸数组 width,height:(N,) 的完整光线追踪"""
    N = verts.shape[0]
    K = rnd.shape[1]
    shadow = np.zeros(N, dtype=np.int64)
    block = np.zeros(N, dtype=np.int64)
    hit = np.zeros(N, dtype=np.int64)
    cx = tower_xy[0]; cy = tower_xy[1]; cz = tower_h
    tan_eps = np.tan(SUN_HALF_ANGLE)
    zhat = np.array([0.0, 0.0, 1.0])
    e1 = np.array([s_dir[1]*zhat[2]-s_dir[2]*zhat[1],
                   s_dir[2]*zhat[0]-s_dir[0]*zhat[2],
                   s_dir[0]*zhat[1]-s_dir[1]*zhat[0]])
    e1n = np.sqrt(e1[0]**2+e1[1]**2+e1[2]**2)
    if e1n < 1e-12:
        e1 = np.array([1.0, 0.0, 0.0])
    else:
        e1 = e1 / e1n
    e2 = np.array([s_dir[1]*e1[2]-s_dir[2]*e1[1],
                   s_dir[2]*e1[0]-s_dir[0]*e1[2],
                   s_dir[0]*e1[1]-s_dir[1]*e1[0]])
    for i in prange(N):
        out = np.empty(max_cells, dtype=np.int64)
        w = width[i]; h = height[i]
        cx0, cy0, cz0 = centers[i,0], centers[i,1], centers[i,2]
        nix, niy, niz = n_norm[i,0], n_norm[i,1], n_norm[i,2]
        uix, uiy, uiz = u_vec[i,0], u_vec[i,1], u_vec[i,2]
        vix, viy, viz = v_vec[i,0], v_vec[i,1], v_vec[i,2]
        for k in range(K):
            a = rnd[i,k,0]*2.0-1.0
            b = rnd[i,k,1]*2.0-1.0
            px = cx0 + (w/2.0)*a*uix + (h/2.0)*b*vix
            py = cy0 + (w/2.0)*a*uiy + (h/2.0)*b*viy
            pz = cz0 + (w/2.0)*a*uiz + (h/2.0)*b*viz
            rho = np.sqrt(rnd[i,k,2]) * tan_eps
            phi = rnd[i,k,3] * 2.0*np.pi
            sx = s_dir[0] + rho*(np.cos(phi)*e1[0] + np.sin(phi)*e2[0])
            sy = s_dir[1] + rho*(np.cos(phi)*e1[1] + np.sin(phi)*e2[1])
            sz = s_dir[2] + rho*(np.cos(phi)*e1[2] + np.sin(phi)*e2[2])
            sn = np.sqrt(sx*sx+sy*sy+sz*sz)
            sx /= sn; sy /= sn; sz /= sn
            if hit_mirror(px, py, pz, sx, sy, sz, i, verts, ncell, offsets, items,
                          bound, cell_size, out, max_cells):
                shadow[i] += 1
                continue
            sd = sx*nix + sy*niy + sz*niz
            rx = -sx + 2.0*sd*nix
            ry = -sy + 2.0*sd*niy
            rz = -sz + 2.0*sd*niz
            if hit_mirror(px, py, pz, rx, ry, rz, i, verts, ncell, offsets, items,
                          bound, cell_size, out, max_cells):
                block[i] += 1
                continue
            if ray_cylinder(px, py, pz, rx, ry, rz, cx, cy, cz, recv_R, recv_H) < INF:
                hit[i] += 1
    return shadow, block, hit
