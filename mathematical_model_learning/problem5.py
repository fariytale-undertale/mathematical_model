"""
2025 国赛 A题 问题5：5机多弹协同干扰3枚导弹 (逆运动学版)
===========================================================
策略: 逆运动学(选LOS点→反算参数) → 贪婪填充 → 精修 → 输出
核心：对每个(无人机,导弹)，在LOS上采样点，反算可行的(θ,v,t_drop,Δt)
"""

import numpy as np
from scipy.optimize import minimize
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

font_path = r'C:\Windows\Fonts\msyh.ttc'
try:
    chinese_font = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [chinese_font.get_name(), 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
except Exception:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                 'axes.unicode_minus': False, 'figure.dpi': 150})

# ============================================================
# 0. 物理常数
# ============================================================
g = 9.8; smoke_r, v_sink, T_life = 10.0, 3.0, 20.0
O = np.array([0.0, 0.0, 0.0])
T_cy, T_r_val, T_h_val = 200.0, 7.0, 10.0

MISSILES = {
    'M1': {'M0': np.array([20000.0, 0.0, 2000.0])},
    'M2': {'M0': np.array([19000.0, 600.0, 2100.0])},
    'M3': {'M0': np.array([18000.0, -600.0, 1900.0])},
}
v_m = 300.0
for key in MISSILES:
    M0 = MISSILES[key]['M0']
    d = np.linalg.norm(M0 - O)
    MISSILES[key]['v'] = (O - M0) / d * v_m
    MISSILES[key]['flight_time'] = d / v_m

DRONES = {
    'FY1': np.array([17800.0, 0.0, 1800.0]),
    'FY2': np.array([12000.0, 1400.0, 1400.0]),
    'FY3': np.array([6000.0, -3000.0, 700.0]),
    'FY4': np.array([11000.0, 2000.0, 1800.0]),
    'FY5': np.array([13000.0, -2000.0, 1300.0]),
}
DRONE_NAMES = list(DRONES.keys())
MISSILE_NAMES = list(MISSILES.keys())
MAX_BOMBS = 3
MAX_DT = {n: np.sqrt(2 * DRONES[n][2] / g) * 0.97 for n in DRONE_NAMES}

t_start = time.time()
print("=" * 65)
print("问题5：5机多弹协同干扰3枚导弹 (逆运动学版)")
print("=" * 65)
for key in MISSILES:
    v = MISSILES[key]['v']
    print(f"  {key}: M0={MISSILES[key]['M0']}, v=({v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}), "
          f"飞行{MISSILES[key]['flight_time']:.1f}s")
for name in DRONE_NAMES:
    print(f"  {name}: {DRONES[name]}, max Δt<{MAX_DT[name]:.1f}s")

# ============================================================
# 1. 圆柱采样
# ============================================================
def build_cylinder_samples():
    pts = []
    for i in range(16):
        th = 2*np.pi*i/16
        if np.sin(th) >= 0: continue
        for j in range(5):
            pts.append([T_r_val*np.cos(th), T_cy+T_r_val*np.sin(th), T_h_val*j/4])
    for z_val in [0.0, T_h_val]:
        for ir in [1, 2, 3]:
            rho = T_r_val*ir/4
            for j in range(12):
                th = 2*np.pi*j/12
                if np.sin(th) >= 0: continue
                pts.append([rho*np.cos(th), T_cy+rho*np.sin(th), z_val])
    return np.array(pts)

cyl_pts = build_cylinder_samples()
T_pt = np.array([0.0, 200.0, 5.0])
print(f"圆柱采样: {len(cyl_pts)} 点")

# ============================================================
# 2. 逆运动学 + 仿真
# ============================================================

def inverse_kinematics(drone_name, C_target, t_det):
    """
    给定烟雾起爆目标位置C_target和时间t_det，反算无人机参数。
    返回: (theta, v, t_drop, dt_fuze) 或 None (不可行)
    """
    D0 = DRONES[drone_name]
    drone_z = D0[2]

    # 水平位移
    d_xy = C_target[:2] - D0[:2]
    if t_det < 1e-6:
        return None  # t_det must be positive

    # 速度向量 (水平)
    v_vec_xy = d_xy / t_det
    v = np.linalg.norm(v_vec_xy)
    if v < 70 or v > 140:
        return None

    theta = np.arctan2(v_vec_xy[1], v_vec_xy[0])
    if theta < 0:
        theta += 2*np.pi

    # 垂直: C_z = drone_z - 0.5*g*dt²
    dz = drone_z - C_target[2]
    if dz <= 0:
        return None  # target must be below drone
    dt_fuze = np.sqrt(2 * dz / g)
    if dt_fuze < 0.5 or dt_fuze > min(14.0, MAX_DT[drone_name]):
        return None

    t_drop = t_det - dt_fuze
    if t_drop < 0:
        return None

    # 验证: 用正运动学算一遍
    r = sim_one_direct(D0, theta, v, t_drop, dt_fuze)
    if r is None:
        return None
    C_actual, drop_pos = r
    # 检查是否到达目标附近 (允许一些误差, 因为用的是平均速度)
    if np.linalg.norm(C_actual - C_target) > 50:
        return None

    return theta, v, t_drop, dt_fuze, C_actual, drop_pos


def sim_one_direct(start_pos, theta, v, t_fly, dt_fuze):
    """正向运动学"""
    v_vec = np.array([v*np.cos(theta), v*np.sin(theta), 0.0])
    drop_pos = start_pos + v_vec * t_fly
    det_horiz = drop_pos + v_vec * dt_fuze
    det_z = drop_pos[2] - 0.5*g*dt_fuze**2
    if det_z <= 0: return None
    return np.array([det_horiz[0], det_horiz[1], det_z]), drop_pos


def shielding_union_for_missile(grenades, missile_key, target_pts, dt=0.01):
    """遮蔽区间并集"""
    if not grenades: return 0.0, []
    M0 = MISSILES[missile_key]['M0']; vM = MISSILES[missile_key]['v']
    all_td = [g[1] for g in grenades]
    t_min, t_max = min(all_td), max(all_td) + T_life
    t_arr = np.arange(t_min, t_max + dt, dt)
    n_t = len(t_arr)
    if n_t == 0: return 0.0, []
    M_arr = M0 + vM * t_arr[:, None]
    union = np.zeros(n_t, dtype=bool)
    for C_det, t_det in grenades:
        active = (t_arr >= t_det - 1e-12) & (t_arr <= t_det + T_life + 1e-12)
        if not np.any(active): continue
        idx = np.where(active)[0]; t_sub = t_arr[idx]; n_sub = len(t_sub)
        C_arr = np.tile(C_det, (n_sub, 1)); C_arr[:, 2] -= v_sink*(t_sub - t_det)
        M_sub = M_arr[idx]
        blocked = np.ones(n_sub, dtype=bool)
        for pt in target_pts:
            T_arr = np.tile(pt, (n_sub, 1))
            MC = C_arr - M_sub; vv = T_arr - M_sub
            v2 = np.sum(vv*vv, axis=1); v2[v2 < 1e-12] = 1e-12
            s = np.clip(np.sum(MC*vv, axis=1)/v2, 0.0, 1.0)
            d = np.linalg.norm(C_arr - (M_sub + s[:, None]*vv), axis=1)
            blocked &= (d <= smoke_r)
        union[idx] |= blocked
    edges = np.diff(np.concatenate([[False], union, [False]]).astype(int))
    starts = np.where(edges == 1)[0]; ends = np.where(edges == -1)[0]
    total = np.sum((ends - starts)*dt)
    intervals = [(t_arr[s], t_arr[e]) for s, e in zip(starts, ends)]
    return total, intervals


# ============================================================
# 3. 逆运动学采样：LOS网格搜索
# ============================================================
print(f"\n{'='*65}")
print(f"阶段1: 逆运动学LOS采样 (N=5000 每对)")
print(f"{'='*65}")

def sample_los_for_drone_missile(drone_name, missile_key, n_samples=5000):
    """
    在导弹→目标的LOS上采样起爆点，用逆运动学反算可行性。
    返回: 最佳参数, 圆柱时长, 区间
    """
    D0 = DRONES[drone_name]
    M0 = MISSILES[missile_key]['M0']
    vM = MISSILES[missile_key]['v']
    ft = MISSILES[missile_key]['flight_time']

    best_params = None  # (theta, v, t_drop, dt_fuze, C_det, t_det)
    best_dur = 0.0
    best_intv = []

    np.random.seed(hash(drone_name + missile_key + "_los") % 2**31)

    for _ in range(n_samples):
        # 采样t_det: 导弹发射后到命中前
        t_det = np.random.uniform(0.1, ft * 0.95)

        # M在t_det时刻的位置
        M_t = M0 + vM * t_det

        # 在M→T线段上采样C (λ∈[0.05, 0.95])
        lam = np.random.uniform(0.05, 0.95)
        C_target = M_t + lam * (T_pt - M_t)

        # 在C附近加一些扰动 (±15m)
        C_target = C_target + np.random.normal(0, 5, 3)
        C_target[2] = max(1, C_target[2])  # 保持在地面以上

        # 逆运动学
        result = inverse_kinematics(drone_name, C_target, t_det)
        if result is None:
            continue

        theta, v, t_drop, dt_fuze, C_actual, drop_pos = result

        # 计算遮蔽时长 (点近似, 快速)
        dur, intv = shielding_union_for_missile([(C_actual, t_det)], missile_key, [T_pt])

        if dur > best_dur:
            best_dur = dur
            best_params = (theta, v, t_drop, dt_fuze, C_actual, t_det)
            best_intv = intv

    if best_params is None:
        return None, 0.0, []

    # 圆柱验证
    theta, v, t_drop, dt_fuze, C_actual, t_det = best_params
    dur_cyl, intv_cyl = shielding_union_for_missile([(C_actual, t_det)], missile_key, cyl_pts)

    # 局部爬山
    improved = True
    best_theta, best_v, best_td, best_dtf = theta, v, t_drop, dt_fuze
    best_cyl = dur_cyl
    for _ in range(3):
        if not improved: break
        improved = False
        for s in np.linspace(0.95, 1.05, 7):
            for j in range(4):
                nth, nv, ntd, ndtf = best_theta, best_v, best_td, best_dtf
                if j == 0: nth = best_theta * s
                elif j == 1: nv = np.clip(best_v * s, 70, 140)
                elif j == 2: ntd = max(0, best_td * s)
                else: ndtf = np.clip(best_dtf * s, 0.5, min(14, MAX_DT[drone_name]))
                r = sim_one_direct(D0, nth, nv, ntd, ndtf)
                if r is None: continue
                C_new, _ = r; tdet_new = ntd + ndtf
                d, _ = shielding_union_for_missile([(C_new, tdet_new)], missile_key, cyl_pts)
                if d > best_cyl + 0.001:
                    best_cyl = d; best_theta, best_v, best_td, best_dtf = nth, nv, ntd, ndtf
                    improved = True

    r_final = sim_one_direct(D0, best_theta, best_v, best_td, best_dtf)
    if r_final:
        C_final, _ = r_final
        dur_cyl, intv_cyl = shielding_union_for_missile([(C_final, best_td+best_dtf)], missile_key, cyl_pts)
        return (best_theta, best_v, best_td, best_dtf, C_final, best_td+best_dtf), dur_cyl, intv_cyl
    return best_params, dur_cyl, intv_cyl


capability = {}
for dn in DRONE_NAMES:
    for mk in MISSILE_NAMES:
        params, dur, intv = sample_los_for_drone_missile(dn, mk)
        capability[(dn, mk)] = (params, dur, intv)
        s = f"{dur:.3f}s [{intv[0][0]:.1f},{intv[0][1]:.1f}]" if dur > 0 and intv else ("×" if dur == 0 else f"{dur:.3f}s")
        print(f"  {dn}→{mk}: {s}", end="  " if mk != 'M3' else "\n")

print(f"\n  能力矩阵 (圆柱 s):")
print(f"  {'':>6}  {'M1':>8}  {'M2':>8}  {'M3':>8}")
for dn in DRONE_NAMES:
    vals = [f"{capability[(dn,mk)][1]:.3f}" if capability[(dn,mk)][1] > 0 else "     ×" for mk in MISSILE_NAMES]
    print(f"  {dn:>6}  {vals[0]:>8}  {vals[1]:>8}  {vals[2]:>8}")
print(f"  耗时: {time.time()-t_start:.1f}s")

# ============================================================
# 4. 贪婪填充
# ============================================================
print(f"\n{'='*65}")
print(f"阶段2: 贪婪时间窗口填充")
print(f"{'='*65}")

remaining = {dn: MAX_BOMBS for dn in DRONE_NAMES}
missile_plan = {mk: [] for mk in MISSILE_NAMES}  # [(dn, th, v, td, dt, C, tdet)]
missile_grenades = {mk: [] for mk in MISSILE_NAMES}  # [(C, tdet)]


def sample_and_optimize_bomb(drone_name, missile_key, existing, n_samples=5000):
    """用逆运动学+采样为一枚弹找最佳边际贡献"""
    D0 = DRONES[drone_name]
    M0 = MISSILES[missile_key]['M0']; vM = MISSILES[missile_key]['v']
    ft = MISSILES[missile_key]['flight_time']
    cur_dur = shielding_union_for_missile(existing, missile_key, [T_pt])[0] if existing else 0.0

    best_marginal = 0.0
    best_result = None

    np.random.seed(hash(f"{drone_name}_{missile_key}_g_{len(existing)}") % 2**31)

    for _ in range(n_samples):
        t_det = np.random.uniform(0.1, ft * 0.95)
        M_t = M0 + vM * t_det
        lam = np.random.uniform(0.05, 0.95)
        C_target = M_t + lam * (T_pt - M_t) + np.random.normal(0, 5, 3)
        C_target[2] = max(1, C_target[2])

        result = inverse_kinematics(drone_name, C_target, t_det)
        if result is None: continue
        theta, v, t_drop, dt_fuze, C_actual, drop_pos = result

        if np.any(np.isnan(C_actual)):
            continue

        dur_new, _ = shielding_union_for_missile(existing + [(C_actual, t_det)], missile_key, [T_pt])
        marginal = dur_new - cur_dur
        if marginal > best_marginal:
            best_marginal = marginal
            best_result = (theta, v, t_drop, dt_fuze, C_actual, t_det, marginal, dur_new)

    # 局部爬山
    if best_result is not None:
        theta0, v0, td0, dt0, C0, tdet0, _, _ = best_result
        best_th, best_v, best_td, best_dt = theta0, v0, td0, dt0
        best_d = best_result[7]
        improved = True
        for _ in range(2):
            if not improved: break
            improved = False
            for s in np.linspace(0.96, 1.04, 9):
                for j in range(4):
                    nth, nv, ntd, ndt = best_th, best_v, best_td, best_dt
                    if j == 0: nth = best_th * s
                    elif j == 1: nv = np.clip(best_v*s, 70, 140)
                    elif j == 2: ntd = max(0, best_td*s)
                    else: ndt = np.clip(best_dt*s, 0.5, min(14, MAX_DT[drone_name]))
                    r = sim_one_direct(D0, nth, nv, ntd, ndt)
                    if r is None: continue
                    C_new, _ = r; tdet_new = ntd + ndt
                    d, _ = shielding_union_for_missile(existing+[(C_new, tdet_new)], missile_key, [T_pt])
                    if d > best_d + 0.003:
                        best_d = d; best_th, best_v, best_td, best_dt = nth, nv, ntd, ndt
                        improved = True

        r_f = sim_one_direct(D0, best_th, best_v, best_td, best_dt)
        if r_f:
            C_f, _ = r_f; tdet_f = best_td + best_dt
            marginal_f = best_d - cur_dur
            return (best_th, best_v, best_td, best_dt, C_f, tdet_f), marginal_f, best_d

    return None, 0.0, cur_dur


# 对所有导弹贪婪填充
for mk in MISSILE_NAMES:
    print(f"\n  --- {mk} ---")
    ranking = sorted(DRONE_NAMES,
                     key=lambda dn: capability.get((dn, mk), (None, 0, []))[1], reverse=True)
    improved = True; n_rounds = 0
    max_per_missile = 5  # 限制每枚导弹最多5弹, 给其他导弹留资源
    while improved and n_rounds < 15 and len(missile_plan[mk]) < max_per_missile:
        improved = False; n_rounds += 1
        best_m, best_r, best_dn = 0.0, None, None
        for dn in ranking[:4]:
            if remaining[dn] <= 0: continue
            res = sample_and_optimize_bomb(dn, mk, missile_grenades[mk], n_samples=3000)
            if res[0] is not None and res[1] > best_m:
                best_m, best_r, best_dn = res[1], res[0], dn
        if best_m > 0.04:
            th, v, td, dtf, C, tdet = best_r
            missile_plan[mk].append((best_dn, th, v, td, dtf, C, tdet))
            missile_grenades[mk].append((C, tdet))
            remaining[best_dn] -= 1
            improved = True
            dur_now = shielding_union_for_missile(missile_grenades[mk], mk, [T_pt])[0]
            print(f"    + {best_dn}弹{MAX_BOMBS-remaining[best_dn]}: "
                  f"边际+{best_m:.3f}s, 总{dur_now:.3f}s, "
                  f"θ={np.degrees(th):.0f}°, v={v:.0f}, td={td:.2f}, Δt={dtf:.2f}, det@{tdet:.2f}s")
    dur_pt, intv_pt = shielding_union_for_missile(missile_grenades[mk], mk, [T_pt])
    print(f"    累计: {len(missile_plan[mk])}弹, 点近似={dur_pt:.3f}s")
    if intv_pt:
        for ts, te in intv_pt:
            print(f"      [{ts:.2f}, {te:.2f}] ({te-ts:.2f}s)")

# 弹量
print(f"\n  弹量使用:")
for dn in DRONE_NAMES:
    used = MAX_BOMBS - remaining[dn]
    dn_count = sum(1 for mk in MISSILE_NAMES for e in missile_plan[mk] if e[0] == dn)
    print(f"    {dn}: {'█'*dn_count}{'░'*(MAX_BOMBS-dn_count)} {dn_count}/{MAX_BOMBS}")

# ============================================================
# 5. 交叉检查 + M2/M3补充
# ============================================================
print(f"\n{'='*65}")
print(f"阶段2B: 交叉导弹遮蔽检查 + 补充")
print(f"{'='*65}")

for mk in ['M2', 'M3']:
    # 检查已有所有弹对mk的交叉遮蔽
    all_grenades = []
    for mk2 in MISSILE_NAMES:
        all_grenades.extend(missile_grenades[mk2])

    cross_cyl, cross_intv = shielding_union_for_missile(all_grenades, mk, cyl_pts)
    print(f"  已有弹→{mk}: 圆柱={cross_cyl:.3f}s")

    # 将有效交叉遮蔽的弹也注册到mk
    existing_keys = {(tuple(g[0]), g[1]) for g in missile_grenades[mk]}
    for mk2 in MISSILE_NAMES:
        for i, (C, tdet) in enumerate(missile_grenades[mk2]):
            key = (tuple(C), tdet)
            if key not in existing_keys:
                sd, _ = shielding_union_for_missile([(C, tdet)], mk, cyl_pts)
                if sd > 0.1:
                    dn = missile_plan[mk2][i][0]
                    th, v, td, dtf = missile_plan[mk2][i][1:5]
                    missile_plan[mk].append((dn, th, v, td, dtf, C, tdet))
                    missile_grenades[mk].append((C, tdet))
                    existing_keys.add(key)

    dur_cyl, _ = shielding_union_for_missile(missile_grenades[mk], mk, cyl_pts)
    print(f"    → 注册后{mk}: {len(missile_plan[mk])}弹, 圆柱={dur_cyl:.3f}s")

    # 补充独立弹
    if dur_cyl < 3.0:
        print(f"    独立补充...")
        ranking = sorted(DRONE_NAMES,
                         key=lambda dn: capability.get((dn, mk), (None, 0, []))[1], reverse=True)
        improved = True; n_rounds = 0
        while improved and n_rounds < 8:
            improved = False; n_rounds += 1
            best_m, best_r, best_dn = 0.0, None, None
            for dn in ranking[:3]:
                if remaining[dn] <= 0: continue
                res = sample_and_optimize_bomb(dn, mk, missile_grenades[mk], n_samples=4000)
                if res[0] is not None and res[1] > best_m:
                    best_m, best_r, best_dn = res[1], res[0], dn
            if best_m > 0.03:
                th, v, td, dtf, C, tdet = best_r
                missile_plan[mk].append((best_dn, th, v, td, dtf, C, tdet))
                missile_grenades[mk].append((C, tdet))
                remaining[best_dn] -= 1; improved = True
                d_cyl, _ = shielding_union_for_missile(missile_grenades[mk], mk, cyl_pts)
                print(f"      + {best_dn}: 圆柱={d_cyl:.3f}s, "
                      f"θ={np.degrees(th):.0f}°, v={v:.0f}, td={td:.2f}, Δt={dtf:.2f}")

# 更新弹量
print(f"\n  最终弹量:")
for dn in DRONE_NAMES:
    dn_count = sum(1 for mk in MISSILE_NAMES for e in missile_plan[mk] if e[0] == dn)
    # 修正remaining
    remaining[dn] = MAX_BOMBS - dn_count
    print(f"    {dn}: {'█'*dn_count}{'░'*(MAX_BOMBS-dn_count)} {dn_count}/{MAX_BOMBS}")

print(f"  耗时: {time.time()-t_start:.1f}s")

# ============================================================
# 6. 圆柱精修
# ============================================================
print(f"\n{'='*65}")
print(f"阶段3: 圆柱精修")
print(f"{'='*65}")

for mk in MISSILE_NAMES:
    if not missile_grenades[mk]: continue
    dur_cyl, intv_cyl = shielding_union_for_missile(missile_grenades[mk], mk, cyl_pts)
    print(f"  {mk}: {dur_cyl:.3f}s, {len(intv_cyl)}段")

print(f"\n  爬山精修...")
for mk in MISSILE_NAMES:
    if not missile_plan[mk]: continue
    for i in range(len(missile_plan[mk])):
        dn, th0, v0, td0, dt0, C0, tdet0 = missile_plan[mk][i]
        D0 = DRONES[dn]; max_dt = min(14.0, MAX_DT[dn])
        others = [missile_grenades[mk][j] for j in range(len(missile_grenades[mk])) if j != i]
        best_th, best_v, best_td, best_dt = th0, v0, td0, dt0
        best_d = shielding_union_for_missile(others, mk, cyl_pts)[0]

        for s in np.linspace(0.93, 1.07, 11):
            for j in range(4):
                nth, nv, ntd, ndt = best_th, best_v, best_td, best_dt
                if j == 0: nth = best_th * s
                elif j == 1: nv = np.clip(best_v*s, 70, 140)
                elif j == 2: ntd = max(0, best_td*s)
                else: ndt = np.clip(best_dt*s, 0.5, max_dt)
                r = sim_one_direct(D0, nth, nv, ntd, ndt)
                if r is None: continue
                C_new, _ = r; tdet_new = ntd + ndt
                d, _ = shielding_union_for_missile(others+[(C_new, tdet_new)], mk, cyl_pts)
                if d > best_d + 0.002:
                    best_d = d; best_th, best_v, best_td, best_dt = nth, nv, ntd, ndt

        if abs(best_th-th0) > 0.01 or abs(best_v-v0) > 0.5:
            r_new = sim_one_direct(D0, best_th, best_v, best_td, best_dt)
            if r_new:
                C_new, _ = r_new; tdet_new = best_td + best_dt
                missile_plan[mk][i] = (dn, best_th, best_v, best_td, best_dt, C_new, tdet_new)
                missile_grenades[mk][i] = (C_new, tdet_new)

print(f"  耗时: {time.time()-t_start:.1f}s")

# ============================================================
# 7. 最终结果
# ============================================================
print(f"\n{'='*65}")
print(f"最终结果 (dt=2ms圆柱验证)")
print(f"{'='*65}")

dt_fine = 0.002
final_results = {}
for mk in MISSILE_NAMES:
    dur_cyl, intv_cyl = shielding_union_for_missile(missile_grenades[mk], mk, cyl_pts, dt=dt_fine) if missile_grenades[mk] else (0.0, [])
    final_results[mk] = (dur_cyl, intv_cyl)
    print(f"\n  {mk}: {dur_cyl:.4f}s ({len(missile_plan[mk])}弹)")
    for i, (ts, te) in enumerate(intv_cyl, 1):
        print(f"    [{ts:.4f}, {te:.4f}] ({te-ts:.4f}s)")

total_all = sum(final_results[mk][0] for mk in MISSILE_NAMES)
print(f"\n  ★ 三弹总计: {total_all:.4f}s")

# 各弹详情
print(f"\n各弹详情:")
for mk in MISSILE_NAMES:
    if not missile_plan[mk]:
        print(f"  {mk}: 无有效遮蔽")
        continue
    for i, entry in enumerate(missile_plan[mk], 1):
        dn, th, v, td, dtf, C, tdet = entry
        r = sim_one_direct(DRONES[dn], th, v, td, dtf)
        dp = r[1] if r else np.zeros(3)
        sd, _ = shielding_union_for_missile([(C, tdet)], mk, cyl_pts, dt=dt_fine)
        others = [missile_grenades[mk][j] for j in range(len(missile_grenades[mk])) if j != i-1]
        wd = shielding_union_for_missile(missile_grenades[mk], mk, cyl_pts, dt=dt_fine)[0]
        wod = shielding_union_for_missile(others, mk, cyl_pts, dt=dt_fine)[0]
        print(f"  [{mk}] {dn}弹{i}: θ={np.degrees(th):.1f}°, v={v:.1f}, "
              f"td={td:.3f}, Δt={dtf:.3f}, det({C[0]:.0f},{C[1]:.0f},{C[2]:.0f})@{tdet:.3f}s")
        print(f"        投放({dp[0]:.0f},{dp[1]:.0f},{dp[2]:.0f}), 单独={sd:.3f}s, 边际={wd-wod:.3f}s")

# ============================================================
# 8. 保存 result3.xlsx
# ============================================================
print(f"\n{'='*65}")
print(f"保存 result3.xlsx")
print(f"{'='*65}")

import pandas as pd

rows = []
for dn in DRONE_NAMES:
    dn_bombs = []
    for mk in MISSILE_NAMES:
        for entry in missile_plan[mk]:
            if entry[0] == dn:
                dn_bombs.append((mk,) + entry[1:])

    for bi in range(MAX_BOMBS):
        if bi < len(dn_bombs):
            mk, th, v, td, dtf, C, tdet = dn_bombs[bi]
            r = sim_one_direct(DRONES[dn], th, v, td, dtf)
            dp = r[1] if r else np.zeros(3)
            row = {
                '无人机编号': dn,
                '无人机运动方向': round(np.degrees(th), 2),
                '无人机运动速度 (m/s)': round(v, 2),
                '烟幕干扰弹编号': bi + 1,
                '烟幕干扰弹投放点的x坐标 (m)': round(dp[0], 2),
                '烟幕干扰弹投放点的y坐标 (m)': round(dp[1], 2),
                '烟幕干扰弹投放点的z坐标 (m)': round(dp[2], 2),
                '烟幕干扰弹起爆点的x坐标 (m)': round(C[0], 2),
                '烟幕干扰弹起爆点的y坐标 (m)': round(C[1], 2),
                '烟幕干扰弹起爆点的z坐标 (m)': round(C[2], 2),
                '有效干扰时长 (s)': round(final_results[mk][0], 2),
                '干扰的导弹编号': mk,
            }
        else:
            row = {k: '' for k in [
                '无人机编号', '无人机运动方向', '无人机运动速度 (m/s)', '烟幕干扰弹编号',
                '烟幕干扰弹投放点的x坐标 (m)', '烟幕干扰弹投放点的y坐标 (m)',
                '烟幕干扰弹投放点的z坐标 (m)', '烟幕干扰弹起爆点的x坐标 (m)',
                '烟幕干扰弹起爆点的y坐标 (m)', '烟幕干扰弹起爆点的z坐标 (m)',
                '有效干扰时长 (s)', '干扰的导弹编号']}
            row['无人机编号'] = dn
            row['烟幕干扰弹编号'] = bi + 1
        rows.append(row)

rows.append({k: '' for k in rows[0]})
note = {k: '' for k in rows[0]}
note['无人机编号'] = '注：以x轴为正向，逆时针方向为正，取值0~360（度）。'
rows.append(note)

df_out = pd.DataFrame(rows)
df_out.to_excel('result3.xlsx', sheet_name='Sheet1', index=False)
print(f"  保存完成 ({len(rows)-2} 行)")

# ============================================================
# 9. 可视化
# ============================================================
print(f"\n{'='*65}")
print(f"生成可视化")
print(f"{'='*65}")

colors_m = {'M1': '#2196F3', 'M2': '#FF9800', 'M3': '#4CAF50'}
colors_d = {'FY1': '#E63946', 'FY2': '#457B9D', 'FY3': '#2A9D8F',
            'FY4': '#E76F51', 'FY5': '#9B5DE5'}

# 甘特图
fig1, axes1 = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
fig1.suptitle('问题5: 三枚导弹遮蔽区间甘特图', fontsize=14, fontweight='bold')
for idx, mk in enumerate(MISSILE_NAMES):
    ax = axes1[idx]; plan = missile_plan[mk]; dur_tot = final_results[mk][0]
    if not plan:
        ax.text(0.5, 0.5, f'{mk}: 无有效遮蔽', transform=ax.transAxes, ha='center')
        ax.set_ylabel(mk, fontsize=12, fontweight='bold'); continue
    for i, entry in enumerate(plan):
        dn, th, v, td, dtf, C, tdet = entry
        sd, si = shielding_union_for_missile([(C, tdet)], mk, cyl_pts, dt=dt_fine)
        for ts, te in si:
            ax.barh(i, te-ts, left=ts, height=0.5, color=colors_d.get(dn, 'gray'),
                    alpha=0.85, edgecolor='white', linewidth=0.5)
    for ts, te in final_results[mk][1]:
        ax.barh(len(plan), te-ts, left=ts, height=0.5, color='#E91E63', alpha=0.5,
                edgecolor='white', linewidth=0.5)
    ax.set_yticks(list(range(len(plan))) + [len(plan)])
    ax.set_yticklabels([f'{plan[i][0]}' for i in range(len(plan))] + ['并集'], fontsize=8)
    ax.set_title(f'{mk} ({len(plan)}弹, {dur_tot:.3f}s)', fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')
axes1[-1].set_xlabel('时间 (s)', fontsize=11)
fig1.tight_layout()
fig1.savefig('problem5_gantt.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"  甘特图: problem5_gantt.png")

# 轨迹图
fig2 = plt.figure(figsize=(18, 9))
ax_main = fig2.add_axes([0.05, 0.10, 0.72, 0.82])
ax_main.set_title('五机飞行路径与烟幕布设', fontsize=13, fontweight='bold')
for mk in MISSILE_NAMES:
    M0 = MISSILES[mk]['M0']; vM = MISSILES[mk]['v']
    ft = MISSILES[mk]['flight_time']
    traj = np.array([M0+vM*t for t in np.linspace(0, ft, 80)])
    ax_main.plot(traj[:, 0], traj[:, 1], '-', color=colors_m[mk], linewidth=2, alpha=0.6, label=f'{mk}')

for dn in DRONE_NAMES:
    dn_bombs_all = []
    for mk in MISSILE_NAMES:
        for entry in missile_plan[mk]:
            if entry[0] == dn:
                _, th, v, td, dtf, C, tdet = entry
                dn_bombs_all.append((td, th, v, C, dtf, tdet, mk))
    dn_bombs_all.sort(key=lambda x: x[0])
    pos = DRONES[dn]; px, py = [pos[0]], [pos[1]]; cur = pos.copy()
    for td, th, v, C, dtf, tdet, mk in dn_bombs_all:
        vv = np.array([v*np.cos(th), v*np.sin(th), 0]); dp = cur + vv*td
        px.append(dp[0]); py.append(dp[1]); cur = dp
    ax_main.plot(px, py, '-', color=colors_d[dn], linewidth=2, alpha=0.85, label=dn, zorder=5)
    ax_main.scatter([pos[0]], [pos[1]], c=colors_d[dn], s=80, marker='s', zorder=8,
                    edgecolors='black', linewidth=1)
    for td, th, v, C, dtf, tdet, mk in dn_bombs_all:
        ax_main.scatter([C[0]], [C[1]], c=colors_m.get(mk, 'gray'), s=100, marker='*',
                        zorder=12, edgecolors='black', linewidth=1.2)

ax_main.scatter([0], [0], c='gray', s=200, marker='s', zorder=10, edgecolors='black', linewidth=2, label='假目标')
ax_main.scatter([0], [200], c='green', s=200, marker='s', zorder=10, edgecolors='darkgreen', linewidth=2, label='真目标')
all_x = [DRONES[dn][0] for dn in DRONE_NAMES] + [MISSILES[mk]['M0'][0] for mk in MISSILE_NAMES]
all_y = [DRONES[dn][1] for dn in DRONE_NAMES] + [MISSILES[mk]['M0'][1] for mk in MISSILE_NAMES]
# 加入投弹点和起爆点
for dn in DRONE_NAMES:
    for mk in MISSILE_NAMES:
        for entry in missile_plan[mk]:
            if entry[0] == dn:
                _, th, v, td, dtf, C, tdet = entry
                D0 = DRONES[dn]
                r = sim_one_direct(D0, th, v, td, dtf)
                if r:
                    dp = r[1]
                    all_x.extend([dp[0], C[0]])
                    all_y.extend([dp[1], C[1]])
x_margin = max(1500, (max(all_x)-min(all_x))*0.05)
y_margin = max(1500, (max(all_y)-min(all_y))*0.08)
ax_main.set_xlim(min(all_x)-x_margin, max(all_x)+x_margin)
ax_main.set_ylim(min(all_y)-y_margin, max(all_y)+y_margin)
ax_main.set_xlabel('x (m)', fontsize=11); ax_main.set_ylabel('y (m)', fontsize=11)
ax_main.legend(loc='upper right', fontsize=7, ncol=2, framealpha=0.8)
ax_main.grid(True, alpha=0.3, linestyle='--'); ax_main.set_aspect('equal')

ax_leg = fig2.add_axes([0.80, 0.10, 0.18, 0.82]); ax_leg.axis('off')
lt = f"总遮蔽: {total_all:.2f}s\n"
for mk in MISSILE_NAMES:
    lt += f"  {mk}: {final_results[mk][0]:.2f}s ({len(missile_plan[mk])}弹)\n"
lt += f"\n弹量:\n"
for dn in DRONE_NAMES:
    dn_c = sum(1 for mk in MISSILE_NAMES for e in missile_plan[mk] if e[0] == dn)
    lt += f"  {dn}: {dn_c}/{MAX_BOMBS}\n"
ax_leg.text(0.05, 0.95, lt, transform=ax_leg.transAxes, fontsize=8, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#F8F9F9', alpha=0.95, edgecolor='#BDC3C7'))
fig2.suptitle('问题5: 全局轨迹与烟幕布设', fontsize=14, fontweight='bold', y=0.97)
fig2.savefig('problem5_trajectory.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"  轨迹图: problem5_trajectory.png")

# 对比图
fig3, ax3 = plt.subplots(figsize=(10, 6))
durs = [final_results[mk][0] for mk in MISSILE_NAMES]
bars = ax3.bar(MISSILE_NAMES, durs, color=[colors_m[m] for m in MISSILE_NAMES],
               edgecolor='black', linewidth=1.2, alpha=0.85)
for b, d in zip(bars, durs):
    ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.1, f'{d:.3f}s', ha='center', fontsize=12, fontweight='bold')
ax3.axhline(y=total_all/3, color='gray', linestyle='--', alpha=0.5, label=f'均值 {total_all/3:.3f}s')
ax3.set_ylabel('有效遮蔽时长 (s)', fontsize=12)
ax3.set_title(f'三枚导弹遮蔽时长对比 (总计 {total_all:.3f}s)', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10); ax3.grid(True, alpha=0.3, axis='y')
fig3.tight_layout()
fig3.savefig('problem5_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"  对比图: problem5_comparison.png")

print(f"\n{'='*65}")
print(f"全部完成! 总耗时 {time.time()-t_start:.1f}s")
print(f"{'='*65}")
