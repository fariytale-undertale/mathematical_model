"""
2025 国赛 A题 问题1：单弹单目标有效遮蔽时长计算
======================================================
场景：FY1 投放 1 枚烟幕干扰弹，干扰 M1 导弹

已知条件（题目给定）：
  - FY1: 以 120 m/s 朝向假目标(原点)飞行，高度 1800m
  - 受领任务 1.5s 后投放，间隔 3.6s 起爆
  - M1: 速度 300 m/s，飞向假目标(原点)
  - 真目标: 圆柱 r=7m, h=10m, 底面圆心 (0,200,0)
  - 烟幕: 球体 r=10m, 起爆后 3 m/s 下沉, 有效 20s

简化假设：真目标距离很远 → 近似为点 T=(0, 200, 5)
方法：计算 M1→T 视线与烟幕球心的距离，≤10m 即为有效遮蔽
      对时间积分得到遮蔽总时长
"""

import numpy as np

# ============================================================
# 0. 物理常数
# ============================================================
g = 9.8          # 重力加速度 (m/s²)

# ============================================================
# 1. 初始条件
# ============================================================
# 假目标（诱饵，原点）
O = np.array([0.0, 0.0, 0.0])

# 真目标 — 近似为点 (圆柱中心)
T = np.array([0.0, 200.0, 5.0])

# 导弹 M1
M1_0 = np.array([20000.0, 0.0, 2000.0])
v_missile = 300.0                                  # m/s

# 无人机 FY1
FY1_0 = np.array([17800.0, 0.0, 1800.0])
v_fy1 = 120.0                                      # m/s

# 投放与起爆
t_drop = 1.5        # 受领任务后 1.5s 投放
dt_fuze = 3.6       # 投放后 3.6s 起爆（引信延时）
t_det = t_drop + dt_fuze  # 起爆时刻 = 5.1s

# 烟幕
smoke_r = 10.0      # 有效半径 (m)
v_sink = 3.0        # 下沉速度 (m/s)
T_effective = 20.0  # 有效时长 (s)

# ============================================================
# 2. 运动学计算
# ============================================================

# ---- 导弹方向 ----
dir_M1 = O - M1_0
dist_M1 = np.linalg.norm(dir_M1)
u_M1 = dir_M1 / dist_M1
v_M1 = u_M1 * v_missile

# ---- FY1 方向（朝向原点）----
# 在 xy 平面内朝向原点，z 不变（等高度飞行）
dir_FY1_xy = np.array([O[0] - FY1_0[0], O[1] - FY1_0[1], 0.0])
dist_FY1_xy = np.linalg.norm(dir_FY1_xy)
u_FY1 = dir_FY1_xy / dist_FY1_xy
v_FY1 = u_FY1 * v_fy1

# ---- 投放时刻 FY1 位置 ----
FY1_drop = FY1_0 + v_FY1 * t_drop

# ---- 起爆时刻烟幕中心位置 ----
# 投放后炸弹在重力作用下运动：
#   水平保持投放时无人机速度，垂直自由落体
bomb_v_horiz = v_FY1  # 水平速度 = 无人机速度
bomb_pos_det_horiz = FY1_drop + bomb_v_horiz * dt_fuze  # 水平位置
bomb_pos_det_z = FY1_drop[2] - 0.5 * g * dt_fuze**2     # 垂直位置
C_det = np.array([bomb_pos_det_horiz[0],
                   bomb_pos_det_horiz[1],
                   bomb_pos_det_z])

print("=" * 60)
print("问题1：有效遮蔽时长计算")
print("=" * 60)
print(f"\n投放时刻: t = {t_drop:.1f}s")
print(f"  无人机位置: ({FY1_drop[0]:.1f}, {FY1_drop[1]:.1f}, {FY1_drop[2]:.1f})")
print(f"\n起爆时刻: t = {t_det:.1f}s")
print(f"  烟幕中心: ({C_det[0]:.1f}, {C_det[1]:.1f}, {C_det[2]:.1f})")
print(f"  导弹位置: ({M1_0[0] + v_M1[0]*t_det:.1f}, "
      f"{M1_0[1] + v_M1[1]*t_det:.1f}, "
      f"{M1_0[2] + v_M1[2]*t_det:.1f})")

# ============================================================
# 3. 遮蔽判定（时间离散化）
# ============================================================
dt = 0.001                       # 时间步长 (s) — 1ms 精度
t_start = t_det                  # 起爆时刻
t_end = t_det + T_effective      # 烟幕失效时刻

n_steps = int((t_end - t_start) / dt)
print(f"\n时间步长: {dt*1000:.0f}ms, 共 {n_steps} 步")

# 预分配
shielded = np.zeros(n_steps, dtype=bool)
distances = np.zeros(n_steps)

for i in range(n_steps):
    t = t_start + i * dt

    # ---- 导弹位置 ----
    M1_t = M1_0 + v_M1 * t

    # ---- 烟幕球心位置 ----
    C_t = C_det.copy()
    C_t[2] -= v_sink * (t - t_det)  # 只有 z 下沉

    # ---- 视线遮蔽判定 ----
    # 球心 C 到线段 M1→T 的最短距离
    MC = C_t - M1_t
    v_line = T - M1_t
    v_norm_sq = np.dot(v_line, v_line)

    if v_norm_sq < 1e-6:
        d = np.linalg.norm(MC)  # 导弹已在目标处
    else:
        # 投影参数 s = (C-M)·(T-M) / |T-M|²
        s = np.dot(MC, v_line) / v_norm_sq

        if s < 0:
            # 最近点在线段后方 → 距离 = |C - M1|
            d = np.linalg.norm(MC)
        elif s > 1:
            # 最近点在线段前方 → 距离 = |C - T|
            d = np.linalg.norm(C_t - T)
        else:
            # 最近点在线段上 → 垂线距离
            cross = np.cross(MC, v_line)
            d = np.linalg.norm(cross) / np.sqrt(v_norm_sq)

    distances[i] = d
    shielded[i] = (d <= smoke_r)

# ============================================================
# 4. 统计有效遮蔽时间
# ============================================================
# 方法：找到所有连续的 True 区间，求和
# 用差分找遮蔽区间的起止位置
is_shielded_int = shielded.astype(int)
edges = np.diff(is_shielded_int, prepend=0, append=0)
start_indices = np.where(edges == 1)[0]   # 0→1 上升沿
end_indices = np.where(edges == -1)[0]    # 1→0 下降沿

intervals = []
total_time = 0.0
for si, ei in zip(start_indices, end_indices):
    t_interval_start = t_start + si * dt
    t_interval_end = t_start + ei * dt
    duration = t_interval_end - t_interval_start
    intervals.append((t_interval_start, t_interval_end, duration))
    total_time += duration

# ============================================================
# 5. 输出结果
# ============================================================
print(f"\n{'='*60}")
print("计算结果")
print(f"{'='*60}")

if len(intervals) == 0:
    print("未找到任何有效遮蔽时段！")
else:
    print(f"\n共 {len(intervals)} 段连续遮蔽区间：")
    print(f"{'序号':<6}{'起始时刻(s)':<16}{'结束时刻(s)':<16}{'持续时长(s)':<14}")
    print("-" * 52)
    for idx, (ts, te, dur) in enumerate(intervals, 1):
        print(f"{idx:<6}{ts:<16.6f}{te:<16.6f}{dur:<14.6f}")

print(f"\n{'─'*52}")
print(f"  总有效遮蔽时长: {total_time:.6f} s")
print(f"{'─'*52}")

# ============================================================
# 6. 补充信息
# ============================================================
print(f"\n补充信息：")
print(f"  导弹飞行时间到原点: {dist_M1 / v_missile:.2f} s")
print(f"  烟幕有效窗口: [{t_det:.1f}, {t_det + T_effective:.1f}] s")
print(f"  起爆时导弹-真目标距离: {np.linalg.norm(M1_0 + v_M1 * t_det - T):.1f} m")
print(f"  起爆时烟幕-真目标距离: {np.linalg.norm(C_det - T):.1f} m")

# 遮蔽距离的最小值
min_d = distances.min()
min_d_idx = np.argmin(distances)
min_d_time = t_start + min_d_idx * dt
print(f"  视线到烟幕中心最小距离: {min_d:.3f} m (t={min_d_time:.3f}s)")

# 遮蔽比例
shield_ratio = total_time / T_effective * 100
print(f"  烟幕有效期内遮蔽比例: {shield_ratio:.2f}%")

# ============================================================
# 7. 改进方法对比
# ============================================================
print(f"\n{'='*60}")
print("改进方法对比")
print(f"{'='*60}")

# ---- 7a. 改进1：圆柱体多点采样 ----
def point_to_segment_distance(P, A, B):
    """点 P 到线段 AB 的最短距离"""
    AP = P - A
    AB = B - A
    ab2 = np.dot(AB, AB)
    if ab2 < 1e-12:
        return np.linalg.norm(AP)
    s = np.clip(np.dot(AP, AB) / ab2, 0.0, 1.0)
    closest = A + s * AB
    return np.linalg.norm(P - closest)


def shielding_interval_for_target_point(T_pt, M1_0, v_M1, C_det, v_sink,
                                         smoke_r, t_det, t_end, dt=0.001):
    """对单个目标点，返回 (start, end) 遮蔽区间或 None"""
    n = int((t_end - t_det) / dt)
    in_shield = False
    t_start_shield = None
    for i in range(n):
        t = t_det + i * dt
        M1_t = M1_0 + v_M1 * t
        C_t = C_det.copy()
        C_t[2] -= v_sink * (t - t_det)
        d = point_to_segment_distance(C_t, M1_t, T_pt)
        if d <= smoke_r and not in_shield:
            in_shield = True
            t_start_shield = t
        elif d > smoke_r and in_shield:
            return (t_start_shield, t - dt)
    if in_shield:
        return (t_start_shield, t_end)
    return None


# 在圆柱面上采样
T_r_val = 7.0
T_h_val = 10.0
print(f"\n--- 改进1: 圆柱体多点采样 ---")
n_theta = 20   # 圆周采样
n_z = 5        # 高度采样
sample_points = []
for i in range(n_theta):
    angle = 2 * np.pi * i / n_theta
    for j in range(n_z):
        z = T_h_val * j / (n_z - 1) if n_z > 1 else T_h_val / 2
        pt = np.array([T_r_val * np.cos(angle),
                       200.0 + T_r_val * np.sin(angle), z])
        sample_points.append(pt)

intervals_all = []
for pt in sample_points:
    result = shielding_interval_for_target_point(
        pt, M1_0, v_M1, C_det, v_sink, smoke_r, t_det, t_end, dt=0.001)
    if result:
        intervals_all.append(result)

if intervals_all:
    starts = [iv[0] for iv in intervals_all]
    ends = [iv[1] for iv in intervals_all]
    # 有效遮蔽 = 所有采样点都被遮蔽 → max(start), min(end)
    t_cyl_start = max(starts)
    t_cyl_end = min(ends)
    dur_cyl = max(0, t_cyl_end - t_cyl_start)
    print(f"  采样点数: {len(sample_points)}")
    print(f"  各点遮蔽开始: [{min(starts):.4f}, {max(starts):.4f}] s")
    print(f"  各点遮蔽结束: [{min(ends):.4f}, {max(ends):.4f}] s")
    print(f"  全部遮蔽区间: [{t_cyl_start:.4f}, {t_cyl_end:.4f}] s")
    print(f"  有效遮蔽时长: {dur_cyl:.4f} s")
    print(f"  (与单点近似 {total_time:.4f}s 相差 {abs(dur_cyl - total_time)*1000:.1f}ms)")
else:
    print("  无遮蔽区间")

# ---- 7b. 改进2：解析求根 ----
print(f"\n--- 改进2: 解析求根（四次方程） ---")
# 对于点目标 T，遮蔽条件 d²(t) = r²
# d²(t) = |MC × v|² / |v|²     (当 s ∈ [0,1] 时)
# 其中 M(t) = M0 + vM*t, C(t) = C_det + vS*t, v = T - M(t)
#
# |MC × v|² 和 |v|² 都是 t 的多项式
# 展开后是 t 的四次方程，用 scipy 求根

try:
    from scipy.optimize import root_scalar
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("  (scipy 不可用，跳过)")

if HAS_SCIPY:
    def d_sq_minus_r2(t):
        """遮蔽判据函数 f(t) = d(t)² - r²，f(t) ≤ 0 表示遮蔽"""
        M1_t = M1_0 + v_M1 * t
        C_t = C_det.copy()
        C_t[2] -= v_sink * (t - t_det)

        MC = C_t - M1_t
        v = T - M1_t
        v2 = np.dot(v, v)
        s = np.dot(MC, v) / v2

        if 0 <= s <= 1:
            cross = np.cross(MC, v)
            d2 = np.dot(cross, cross) / v2
        elif s < 0:
            d2 = np.dot(MC, MC)
        else:
            d2 = np.dot(C_t - T, C_t - T)

        return d2 - smoke_r**2

    # 在烟幕有效窗口内搜索根
    # 先用粗步长找正负号变化区间
    t_scan = np.linspace(t_det, t_end, 500)
    f_vals = np.array([d_sq_minus_r2(t) for t in t_scan])

    roots_found = []
    for i in range(len(t_scan) - 1):
        if f_vals[i] * f_vals[i+1] < 0:  # 符号变化
            try:
                sol = root_scalar(d_sq_minus_r2,
                                  bracket=[t_scan[i], t_scan[i+1]],
                                  method='brentq', xtol=1e-12)
                roots_found.append(sol.root)
            except Exception:
                pass
        elif abs(f_vals[i]) < 1e-6:
            roots_found.append(t_scan[i])

    # 去重
    roots_found = sorted(set(round(r, 8) for r in roots_found))
    print(f"  找到 {len(roots_found)} 个根: {[f'{r:.6f}' for r in roots_found]}")

    if len(roots_found) >= 2:
        # 在每对根之间检查是否遮蔽
        for i in range(0, len(roots_found) - 1, 1):
            t_mid = (roots_found[i] + roots_found[i+1]) / 2
            if d_sq_minus_r2(t_mid) < 0:
                t_a, t_b = roots_found[i], roots_found[i+1]
                dur = t_b - t_a
                print(f"  遮蔽区间: [{t_a:.8f}, {t_b:.8f}], 持续 {dur:.8f} s")
                print(f"  (与离散法 {total_time:.6f}s 相差 {abs(dur-total_time)*1e6:.2f}μs)")
