"""
2025 国赛 A题 问题3：单机3弹最优投放策略
==========================================
FY1 投放 3 枚烟幕干扰弹，干扰 M1 导弹。
每枚弹投放前可调整航向和速度，间隔 ≥1s。

优化策略：直接12维联合 DE + 多轮精修
  阶段1: 以问题2最优解为锚，生成多样化初始种群
  阶段2: 大规模 DE 全局搜索 (点近似)
  阶段3: 缩小范围 DE + 圆柱验证
  阶段4: Nelder-Mead 局部精修
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# ========== 中文字体 ==========
font_path = r'C:\Windows\Fonts\msyh.ttc'
try:
    chinese_font = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [chinese_font.get_name(), 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
except Exception:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
rcParams.update({
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False, 'figure.dpi': 150,
})

# ============================================================
# 0. 物理常数
# ============================================================
g = 9.8
smoke_r, v_sink, T_life = 10.0, 3.0, 20.0
O  = np.array([0.0, 0.0, 0.0])
T_cy, T_cz, T_r_val, T_h_val = 200.0, 0.0, 7.0, 10.0
M1_0   = np.array([20000.0, 0.0, 2000.0])
v_m    = 300.0
v_M1   = (O - M1_0) / np.linalg.norm(O - M1_0) * v_m
FY1_0  = np.array([17800.0, 0.0, 1800.0])

# 问题2最优
P2_THETA  = np.radians(7.394)
P2_V      = 98.51
P2_T_DROP = 0.0146
P2_DT     = 0.8811

print(f"v_M1 = ({v_M1[0]:.2f}, {v_M1[1]:.2f}, {v_M1[2]:.2f})")
print(f"M1飞行时间到原点: {np.linalg.norm(M1_0-O)/v_m:.1f}s")

# ============================================================
# 1. 圆柱采样点
# ============================================================
def build_cylinder_samples():
    pts = []
    for i in range(16):
        th = 2 * np.pi * i / 16
        if np.sin(th) >= 0:
            continue
        for j in range(5):
            z = T_h_val * j / 4
            pts.append([T_r_val * np.cos(th), T_cy + T_r_val * np.sin(th), z])
    for z_val in [0.0, T_h_val]:
        for ir in [1, 2, 3]:
            rho = T_r_val * ir / 4
            for j in range(12):
                th = 2 * np.pi * j / 12
                if np.sin(th) >= 0:
                    continue
                pts.append([rho * np.cos(th), T_cy + rho * np.sin(th), z_val])
    return np.array(pts)

cyl_pts = build_cylinder_samples()
T_pt   = np.array([0.0, 200.0, 5.0])
print(f"圆柱采样: {len(cyl_pts)} 点")

# ============================================================
# 2. 核心：运动学 + 遮蔽并集
# ============================================================

def sim_one(start_pos, theta, v, t_fly, dt_fuze):
    """
    从 start_pos 以 (θ, v) 飞行 t_fly 秒后投弹，dt_fuze 后起爆。
    返回: (det_pos, drop_pos) 或 None (触地)
    """
    v_vec = np.array([v * np.cos(theta), v * np.sin(theta), 0.0])
    drop_pos = start_pos + v_vec * t_fly
    det_horiz = drop_pos + v_vec * dt_fuze
    det_z = drop_pos[2] - 0.5 * g * dt_fuze**2
    if det_z <= 0:
        return None
    return np.array([det_horiz[0], det_horiz[1], det_z]), drop_pos


def simulate_three(x):
    """
    12维 → 3枚弹全流程。
    x = [θ₁,v₁,t_drop1,Δt₁, θ₂,v₂,gap12,Δt₂, θ₃,v₃,gap23,Δt₃]
    返回: (grenades_list, drop_positions, det_positions, det_times) 或 None
      其中 grenades_list = [(C_det, t_det), ...]
    """
    th1, v1, t1, dt1 = x[0], x[1], x[2], x[3]
    th2, v2, g12, dt2 = x[4], x[5], x[6], x[7]
    th3, v3, g23, dt3 = x[8], x[9], x[10], x[11]

    r1 = sim_one(FY1_0, th1, v1, t1, dt1)
    if r1 is None: return None
    C1, P1 = r1
    td1 = t1 + dt1

    r2 = sim_one(P1, th2, v2, g12, dt2)
    if r2 is None: return None
    C2, P2 = r2
    td2 = t1 + g12 + dt2

    r3 = sim_one(P2, th3, v3, g23, dt3)
    if r3 is None: return None
    C3, P3 = r3
    td3 = t1 + g12 + g23 + dt3

    return ([(C1, td1), (C2, td2), (C3, td3)],
            [P1, P2, P3], [C1, C2, C3], [td1, td2, td3])


def shielding_union(grenades, target_pts, dt=0.01):
    """
    多弹遮蔽区间并集。
    grenades: [(C_det, t_det), ...]
    返回: (总时长, [(start, end), ...])
    """
    if not grenades:
        return 0.0, []
    all_td = [g[1] for g in grenades]
    t_min, t_max = min(all_td), max(all_td) + T_life
    t_arr = np.arange(t_min, t_max + dt, dt)
    n_t = len(t_arr)
    if n_t == 0:
        return 0.0, []

    M1_arr = M1_0 + v_M1 * t_arr[:, None]
    union = np.zeros(n_t, dtype=bool)

    for C_det, t_det in grenades:
        active = (t_arr >= t_det - 1e-12) & (t_arr <= t_det + T_life + 1e-12)
        if not np.any(active):
            continue
        idx = np.where(active)[0]
        t_sub = t_arr[idx]
        n_sub = len(t_sub)

        C_arr = np.tile(C_det, (n_sub, 1))
        C_arr[:, 2] -= v_sink * (t_sub - t_det)
        M_sub = M1_arr[idx]

        blocked = np.ones(n_sub, dtype=bool)
        for pt in target_pts:
            T_arr = np.tile(pt, (n_sub, 1))
            MC = C_arr - M_sub
            vv = T_arr - M_sub
            v2 = np.sum(vv * vv, axis=1)
            v2[v2 < 1e-12] = 1e-12
            s = np.clip(np.sum(MC * vv, axis=1) / v2, 0.0, 1.0)
            closest = M_sub + s[:, None] * vv
            d = np.linalg.norm(C_arr - closest, axis=1)
            blocked &= (d <= smoke_r)
        union[idx] |= blocked

    edges = np.diff(np.concatenate([[False], union, [False]]).astype(int))
    starts = np.where(edges == 1)[0]
    ends   = np.where(edges == -1)[0]
    total  = np.sum((ends - starts) * dt)
    intervals = [(t_arr[s], t_arr[e]) for s, e in zip(starts, ends)]
    return total, intervals


# ============================================================
# 3. 目标函数
# ============================================================

def objective(x, target_pts):
    """12维目标函数 (返回负时长供DE最小化)"""
    r = simulate_three(x)
    if r is None:
        return 0.0
    dur, _ = shielding_union(r[0], target_pts)
    return -dur


# 快速验证
x_p2 = [P2_THETA, P2_V, P2_T_DROP, P2_DT,
        P2_THETA, P2_V, 5.0, P2_DT,
        P2_THETA, P2_V, 5.0, P2_DT]
r_test = simulate_three(x_p2)
dur_test, intv_test = shielding_union(r_test[0], [T_pt])
print(f"\n问题2参数×3验证: {dur_test:.4f}s (预期 ≈4.82s, 弹2/3浪费)")

# ============================================================
# 4. 阶段1: 大规模随机采样 → 构建初始种群
# ============================================================
print(f"\n{'='*65}")
print(f"阶段1: 随机采样 20000 点，构建初始种群")
print(f"{'='*65}")

# 变量: θ₁,v₁,t1,Δt₁, θ₂,v₂,g12,Δt₂, θ₃,v₃,g23,Δt₃
base_bounds = [
    (0, 2*np.pi), (70, 140), (0, 12), (0.5, 14),      # 弹1
    (0, 2*np.pi), (70, 140), (1, 20), (0.5, 14),      # 弹2
    (0, 2*np.pi), (70, 140), (1, 20), (0.5, 14),      # 弹3
]

np.random.seed(42)
t_samp = time.time()

# 策略: 混合采样 — 50%问题2锚定 + 50%全空间探索
samples = []
n_total = 30000
for k in range(n_total):
    x = np.zeros(12)
    if k < n_total // 2:
        # 锚定策略: 弹1在问题2最优附近, 弹2弹3物理引导
        x[0] = P2_THETA + np.random.normal(0, 0.15)
        x[1] = P2_V + np.random.normal(0, 8)
        x[2] = max(0, P2_T_DROP + np.random.normal(0, 0.3))
        x[3] = max(0.5, P2_DT + np.random.normal(0, 0.5))
        # 弹2: 偏向目标方向 (θ≈π即180°, 飞向-x)
        x[4] = np.random.uniform(np.pi - 1.0, np.pi + 1.0)
        x[5] = np.random.uniform(80, 140)
        x[6] = np.random.uniform(1, 12)
        x[7] = np.random.uniform(1, 10)
        # 弹3: 偏向目标方向
        x[8] = np.random.uniform(np.pi - 1.0, np.pi + 1.0)
        x[9] = np.random.uniform(80, 140)
        x[10] = np.random.uniform(1, 15)
        x[11] = np.random.uniform(1, 10)
    else:
        # 全空间探索
        for j in range(12):
            x[j] = np.random.uniform(*base_bounds[j])
    # 裁剪
    for j in range(12):
        x[j] = np.clip(x[j], base_bounds[j][0], base_bounds[j][1])
    samples.append(x)

# 批量评估 (只用点近似)
print(f"  评估 {n_total} 个样本...")
results = []
for i, x in enumerate(samples):
    r = simulate_three(x)
    if r is None:
        continue
    dur, _ = shielding_union(r[0], [T_pt])
    if dur > 0.5:
        results.append((dur, np.array(x)))
    if (i+1) % 5000 == 0:
        print(f"    {i+1}/{n_total}...")

results.sort(key=lambda v: -v[0])
n_valid = len(results)
print(f"  有效样本: {n_valid}")
if n_valid > 0:
    print(f"  Top5: {[f'{r[0]:.3f}s' for r in results[:5]]}")
    print(f"  Top1 参数预览: θ₁={np.degrees(results[0][1][0]):.1f}°, "
          f"v₁={results[0][1][1]:.1f}, t1={results[0][1][2]:.3f}, Δt₁={results[0][1][3]:.3f}")
print(f"  耗时: {time.time()-t_samp:.1f}s")

# ============================================================
# 5. 阶段2: 大规模 DE（点近似）
# ============================================================
print(f"\n{'='*65}")
print(f"阶段2: DE全局搜索 (50个体, 300代, 点近似)")
print(f"{'='*65}")

# 构建初始种群
popsize = 60
init_pop = []

# Top 25 采样结果 + 多样化个体
if n_valid >= 25:
    for i in range(min(25, n_valid)):
        x = results[i][1] + np.random.normal(0, 0.03, 12) * [
            0.15, 3, 0.15, 0.2, 0.3, 4, 0.5, 0.5, 0.3, 4, 0.5, 0.5]
        init_pop.append(np.clip(x, [b[0] for b in base_bounds], [b[1] for b in base_bounds]))
else:
    for i in range(min(n_valid, popsize)):
        init_pop.append(results[i][1])

# 补充: 包含物理多样化个体
while len(init_pop) < popsize:
    x = np.zeros(12)
    rtype = np.random.random()
    if rtype < 0.4:
        # 问题2锚定
        x[0] = P2_THETA + np.random.uniform(-0.3, 0.3)
        x[1] = np.random.uniform(85, 130)
        x[2] = np.random.uniform(0, 3)
        x[3] = np.random.uniform(0.5, 3)
        x[4] = np.random.uniform(np.pi - 1.0, np.pi + 1.0)
        x[5] = np.random.uniform(80, 140)
        x[6] = np.random.uniform(1, 10)
        x[7] = np.random.uniform(1, 10)
        x[8] = np.random.uniform(np.pi - 1.0, np.pi + 1.0)
        x[9] = np.random.uniform(80, 140)
        x[10] = np.random.uniform(1, 15)
        x[11] = np.random.uniform(1, 10)
    else:
        # 全随机
        for j in range(12):
            x[j] = np.random.uniform(*base_bounds[j])
    x = np.clip(x, [b[0] for b in base_bounds], [b[1] for b in base_bounds])
    init_pop.append(x)

print(f"  初始种群: {len(init_pop)} 个体")
if n_valid > 0:
    init_fitness = [-objective(x, [T_pt]) for x in init_pop[:10]]
    print(f"  前10个体适应度: {[f'{f:.3f}' for f in init_fitness]}")

t_de1 = time.time()
# 收敛跟踪
de1_history = {'best': [], 'mean': []}

def cb_de1(xk, convergence):
    de1_history['best'].append(-objective(xk, [T_pt]))
    if len(de1_history['best']) % 50 == 0:
        print(f"    代{len(de1_history['best'])}: best={de1_history['best'][-1]:.3f}s")

result_de1 = differential_evolution(
    lambda x: objective(x, [T_pt]),
    base_bounds,
    strategy='best1bin',
    maxiter=400,
    popsize=popsize,
    tol=0.00001,
    mutation=(0.5, 1.8),
    recombination=0.7,
    seed=42,
    init=np.array(init_pop),
    polish=False,
    callback=cb_de1,
)
dur_de1 = -result_de1.fun
print(f"\n  DE结果: {dur_de1:.4f}s (点近似)")
print(f"  代数: {result_de1.nit}, 评估: {result_de1.nfev}")
print(f"  耗时: {time.time()-t_de1:.1f}s")

# 验证
r_de1 = simulate_three(result_de1.x)
dur_de1_cyl, intv_de1_cyl = shielding_union(r_de1[0], cyl_pts)
print(f"  圆柱验证: {dur_de1_cyl:.4f}s")

# ============================================================
# 6. 阶段3: Nelder-Mead 圆柱精修 + 网格 (跳过高成本圆柱DE)
# ============================================================
print(f"\n{'='*65}")
print(f"阶段3: NM+网格圆柱精修 (快速)")
print(f"{'='*65}")

x_current = result_de1.x
dur_current = dur_de1_cyl

# 6a. Nelder-Mead 从DE1结果出发, 直接用圆柱采样精修
print(f"  NM精修 (圆柱)...")
try:
    nm = minimize(
        lambda x: objective(x, cyl_pts),
        x_current,
        method='Nelder-Mead',
        options={'xatol': 1e-5, 'fatol': 1e-6, 'maxiter': 800}
    )
    dur_nm = -nm.fun
    if dur_nm > dur_current:
        print(f"  NM: {dur_current:.4f} → {dur_nm:.4f}s (+{dur_nm-dur_current:.4f}s)")
        x_best = nm.x
    else:
        print(f"  NM未提升, 保持DE1结果")
        x_best = x_current
        dur_nm = dur_current
except Exception as e:
    print(f"  NM失败: {e}, 保持DE1结果")
    x_best = x_current
    dur_nm = dur_current

# 6b. 网格精修
print(f"  网格精修...")
best_dur = dur_nm
for scale in np.linspace(0.97, 1.03, 5):
    for j in range(12):
        xt = x_best.copy()
        xt[j] *= scale
        xt[j] = np.clip(xt[j], base_bounds[j][0], base_bounds[j][1])
        d = -objective(xt, cyl_pts)
        if d > best_dur:
            best_dur = d
            x_best = xt
            print(f"    网格提升: {best_dur:.4f}s (参数{j})")

dur_best = best_dur

# ============================================================
# 阶段4B: 弹3专项拯救 (固定弹1+弹2, 宽范围搜弹3)
# ============================================================
r_check = simulate_three(x_best)
dur_check, intv_check = shielding_union(r_check[0], cyl_pts)
# 检测弹3是否零贡献
g3_solo, _ = shielding_union([r_check[0][2]], cyl_pts)

if g3_solo < 0.1:
    print(f"\n{'='*65}")
    print(f"阶段4B: 弹3专项优化 (固定弹1+弹2最优, 宽范围DE)")
    print(f"{'='*65}")

    # 固定前8个参数
    x12_fixed = x_best[:8]

    # 弹3的起点（弹2投弹点）
    r12 = simulate_three(x_best)
    P2_fixed = r12[1][1]  # 弹2投弹位置

    bounds_g3_wide = [
        (0, 2*np.pi),     # θ₃: 全范围
        (70.0, 140.0),    # v₃
        (1.0, 25.0),      # gap23: 扩大上限
        (0.5, 14.0),      # Δt₃
    ]

    def obj_g3_only(x4, target_pts):
        """固定弹1弹2, 只优化弹3"""
        th3, v3, g23, dt3 = x4
        # 弹1弹2的数据
        grenades_12 = [(r_check[0][0][0], r_check[0][0][1]),
                        (r_check[0][1][0], r_check[0][1][1])]
        # 弹3
        r3 = sim_one(P2_fixed, th3, v3, g23, dt3)
        if r3 is None:
            return 0.0
        C3, P3 = r3
        td3 = x12_fixed[2] + x12_fixed[6] + g23 + dt3
        dur, _ = shielding_union(grenades_12 + [(C3, td3)], target_pts)
        return -dur

    # 物理引导+随机混合采样
    # 弹3需要在 t>10s 的晚段起作用，烟雾应布设在导弹→目标之间
    # 因此无人机应向 -x 方向飞行（朝向目标），即 θ₃ ≈ π (180°)
    print(f"  混合暖启动: 物理引导 + 随机...")
    np.random.seed(789)
    t_g3 = time.time()
    g3_samples = []

    # 策略A: 物理引导 — 飞向目标方向 (-x)，在不同距离布设烟幕
    for _ in range(5000):
        th3 = np.random.uniform(np.pi - 0.5, np.pi + 0.5)  # 180° ± 30°
        v3  = np.random.uniform(90, 140)
        # 总时间 t_det3 ≈ gap23 + dt3 + 1.2 ≈ 8~14 → gap23+dt3 ≈ 7~13
        g23 = np.random.uniform(2, 8)
        dt3 = np.random.uniform(3, 10)
        x4 = [th3, v3, g23, dt3]
        d = -obj_g3_only(x4, [T_pt])
        g3_samples.append((d, np.array(x4)))

    # 策略B: 纯随机（覆盖所有方向）
    for _ in range(5000):
        x4 = [np.random.uniform(*b) for b in bounds_g3_wide]
        d = -obj_g3_only(x4, [T_pt])
        g3_samples.append((d, np.array(x4)))

    # 只保留比当前好的
    g3_samples = [(d, x) for d, x in g3_samples if d > dur_check]
    g3_samples.sort(key=lambda v: -v[0])
    n_g3 = len(g3_samples)
    best_g3_dur = g3_samples[0][0] if n_g3 else 0
    print(f"  改进解: {n_g3}, 最优: {best_g3_dur:.3f}s (当前={dur_check:.3f}s)")

    if n_g3 > 0 and best_g3_dur > dur_check + 0.01:
        # DE搜索
        init_g3 = []
        for i in range(min(25, n_g3)):
            xp = g3_samples[i][1] + np.random.normal(0, 0.03, 4) * [0.3, 3, 0.5, 0.5]
            init_g3.append(np.clip(xp, [b[0] for b in bounds_g3_wide],
                                   [b[1] for b in bounds_g3_wide]))
        while len(init_g3) < 30:
            # 混合初始化
            if np.random.random() < 0.6:
                th3 = np.random.uniform(np.pi - 0.8, np.pi + 0.8)
                v3 = np.random.uniform(80, 140)
                g23 = np.random.uniform(2, 10)
                dt3 = np.random.uniform(3, 10)
                init_g3.append([th3, v3, g23, dt3])
            else:
                init_g3.append([np.random.uniform(*b) for b in bounds_g3_wide])

        print(f"  DE搜索 (30个体, 200代)...")
        result_g3 = differential_evolution(
            lambda x: obj_g3_only(x, [T_pt]),
            bounds_g3_wide,
            strategy='best1bin', maxiter=200, popsize=30,
            tol=0.0001, mutation=(0.5, 1.5), recombination=0.7,
            seed=789, init=np.array(init_g3), polish=False
        )

        x_new = x_best.copy()
        x_new[8:] = result_g3.x
        dur_new = -objective(x_new, cyl_pts)
        g3_solo_new, g3_intv_new = shielding_union(
            [(simulate_three(x_new)[0][2][0], simulate_three(x_new)[0][2][1])], cyl_pts)
        print(f"  弹3优化: 总={dur_new:.4f}s, 弹3单独={g3_solo_new:.3f}s "
              f"(之前总={dur_best:.4f}s, 弹3单独={g3_solo:.3f}s)")

        if dur_new > dur_best + 0.005:
            improvement = dur_new - dur_best
            x_best = x_new
            dur_best = dur_new
            print(f"  ★ 弹3拯救成功! +{improvement:.4f}s")

        # NM精修
        print(f"  圆柱精修...")
        try:
            nm_g3 = minimize(
                lambda x: obj_g3_only(x, cyl_pts),
                result_g3.x,
                method='Nelder-Mead',
                bounds=bounds_g3_wide,
                options={'xatol': 1e-6, 'fatol': 1e-7, 'maxiter': 800}
            )
            x_new2 = x_best.copy()
            x_new2[8:] = nm_g3.x
            dur_nm2 = -objective(x_new2, cyl_pts)
            if dur_nm2 > dur_best:
                x_best = x_new2
                dur_best = dur_nm2
                print(f"  NM提升: {dur_best:.4f}s")
        except Exception as e:
            print(f"  NM: {e}")
        print(f"  耗时: {time.time()-t_g3:.1f}s")
    else:
        print(f"  弹3无法有效贡献 (可能已覆盖全部可遮蔽窗口)")

# dur_best 已在各阶段中更新为最优值

# ============================================================
# 8. 最终结果提取
# ============================================================
r_final = simulate_three(x_best)
assert r_final is not None
grenades_f, P_f, C_f, Td_f = r_final

(th1_f, v1_f, t1_f, dt1_f,
 th2_f, v2_f, g12_f, dt2_f,
 th3_f, v3_f, g23_f, dt3_f) = x_best

# 精确验证
dt_fine = 0.002
dur_final, intv_final = shielding_union(grenades_f, cyl_pts, dt=dt_fine)
dur_final_pt, intv_final_pt = shielding_union(grenades_f, [T_pt], dt=dt_fine)

print(f"\n{'='*65}")
print(f"最终结果")
print(f"{'='*65}")

print(f"\n  ┌─────────── 最优参数 ───────────┐")
print(f"  │ 弹1: θ={np.degrees(th1_f):.3f}°, v={v1_f:.2f} m/s        │")
print(f"  │      t_drop={t1_f:.4f}s, Δt={dt1_f:.4f}s             │")
print(f"  │      起爆时刻={Td_f[0]:.4f}s                      │")
print(f"  │      投放=({P_f[0][0]:.1f},{P_f[0][1]:.1f},{P_f[0][2]:.1f})")
print(f"  │      起爆=({C_f[0][0]:.1f},{C_f[0][1]:.1f},{C_f[0][2]:.1f})")
print(f"  ├─────────────────────────────────┤")
print(f"  │ 弹2: θ={np.degrees(th2_f):.3f}°, v={v2_f:.2f} m/s        │")
print(f"  │      间隔={g12_f:.4f}s, Δt={dt2_f:.4f}s                │")
print(f"  │      起爆时刻={Td_f[1]:.4f}s                      │")
print(f"  │      投放=({P_f[1][0]:.1f},{P_f[1][1]:.1f},{P_f[1][2]:.1f})")
print(f"  │      起爆=({C_f[1][0]:.1f},{C_f[1][1]:.1f},{C_f[1][2]:.1f})")
print(f"  ├─────────────────────────────────┤")
print(f"  │ 弹3: θ={np.degrees(th3_f):.3f}°, v={v3_f:.2f} m/s        │")
print(f"  │      间隔={g23_f:.4f}s, Δt={dt3_f:.4f}s                │")
print(f"  │      起爆时刻={Td_f[2]:.4f}s                      │")
print(f"  │      投放=({P_f[2][0]:.1f},{P_f[2][1]:.1f},{P_f[2][2]:.1f})")
print(f"  │      起爆=({C_f[2][0]:.1f},{C_f[2][1]:.1f},{C_f[2][2]:.1f})")
print(f"  └─────────────────────────────────┘")

print(f"\n  遮蔽区间 (圆柱, dt={dt_fine*1000:.0f}ms):")
for i, (ts, te) in enumerate(intv_final, 1):
    print(f"    [{ts:.4f}, {te:.4f}]  ({te-ts:.4f}s)")
print(f"\n  ★ 有效遮蔽总时长: {dur_final:.4f}s (圆柱验证)")
print(f"  ★ 点近似:         {dur_final_pt:.4f}s")
print(f"  总耗时: {time.time()-t_samp:.1f}s")

# 各弹贡献
print(f"\n各弹贡献分析:")
for i in range(3):
    sd, si = shielding_union([grenades_f[i]], cyl_pts, dt=dt_fine)
    others = [grenades_f[j] for j in range(3) if j != i]
    wd, _ = shielding_union(others, cyl_pts, dt=dt_fine)
    print(f"  弹{i+1}: 单独={sd:.4f}s, 边际={dur_final-wd:.4f}s, "
          f"区间={[(f'{a:.3f}',f'{b:.3f}') for a,b in si]}")

# ============================================================
# 9. 保存到 result1.xlsx
# ============================================================
print(f"\n{'='*65}")
print(f"保存 result1.xlsx")
print(f"{'='*65}")

import pandas as pd
from openpyxl import load_workbook

# 直接构建数据（避免模板列类型问题）
data = {
    '无人机运动方向': [round(np.degrees(th1_f), 2),
                     round(np.degrees(th2_f), 2),
                     round(np.degrees(th3_f), 2),
                     '',
                     '注：以x轴为正向，逆时针方向为正，取值0~360（度）。'],
    '无人机运动速度 (m/s)': [round(v1_f, 2), round(v2_f, 2), round(v3_f, 2), '', ''],
    '烟幕干扰弹编号': [1, 2, 3, '', ''],
    '烟幕干扰弹投放点的x坐标 (m)': [round(P_f[0][0], 2), round(P_f[1][0], 2), round(P_f[2][0], 2), '', ''],
    '烟幕干扰弹投放点的y坐标 (m)': [round(P_f[0][1], 2), round(P_f[1][1], 2), round(P_f[2][1], 2), '', ''],
    '烟幕干扰弹投放点的z坐标 (m)': [round(P_f[0][2], 2), round(P_f[1][2], 2), round(P_f[2][2], 2), '', ''],
    '烟幕干扰弹起爆点的x坐标 (m)': [round(C_f[0][0], 2), round(C_f[1][0], 2), round(C_f[2][0], 2), '', ''],
    '烟幕干扰弹起爆点的y坐标 (m)': [round(C_f[0][1], 2), round(C_f[1][1], 2), round(C_f[2][1], 2), '', ''],
    '烟幕干扰弹起爆点的z坐标 (m)': [round(C_f[0][2], 2), round(C_f[1][2], 2), round(C_f[2][2], 2), '', ''],
    '有效干扰时长 (s)': [round(dur_final, 2), round(dur_final, 2), round(dur_final, 2), '', ''],
}

df_out = pd.DataFrame(data)
df_out.to_excel('result1.xlsx', sheet_name='Sheet1', index=False)
print(f"  已保存 result1.xlsx")

# 确认
df_check = pd.read_excel('result1.xlsx')
print(f"\n  保存内容确认:")
print(df_check.to_string())

# ============================================================
# 10. 可视化
# ============================================================
t0_viz = time.time()
colors = ['#2196F3', '#FF9800', '#4CAF50']

# --- 图1: 遮蔽区间甘特图 ---
fig1, ax1 = plt.subplots(figsize=(14, 5))
fig1.suptitle('问题3: 三弹遮蔽区间甘特图', fontsize=14, fontweight='bold')

t_gmin = min(Td_f)
t_gmax = max(Td_f) + T_life

for i in range(3):
    sd, si = shielding_union([grenades_f[i]], cyl_pts, dt=dt_fine)
    for ts, te in si:
        ax1.barh(i, te - ts, left=ts, height=0.5, color=colors[i],
                 alpha=0.85, edgecolor='white', linewidth=0.5)

for ts, te in intv_final:
    ax1.barh(3, te - ts, left=ts, height=0.5, color='#E91E63',
             alpha=0.7, edgecolor='white', linewidth=0.5)

for i in range(3):
    ax1.axvline(x=Td_f[i], color=colors[i], linestyle='--', alpha=0.4, linewidth=1)
    ax1.text(Td_f[i], i + 0.4, f'起爆\n{Td_f[i]:.3f}s', fontsize=7,
             ha='center', va='bottom', color=colors[i])

ax1.set_yticks(range(4))
ax1.set_yticklabels(['弹1', '弹2', '弹3', '并集'], fontsize=11)
ax1.set_xlabel('时间 (s)', fontsize=11)
ax1.set_xlim(t_gmin - 1, t_gmax + 1)
ax1.grid(True, alpha=0.3, axis='x')
fig1.tight_layout()
fig1.savefig('problem3_gantt.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"  甘特图: problem3_gantt.png")

# --- 图2: 轨迹俯视图（双面板：放大+全局）---
fig2 = plt.figure(figsize=(18, 8))

# 面板A: 全局视图（右侧小图）
ax_global = fig2.add_axes([0.62, 0.15, 0.35, 0.72])
ax_global.set_title('全局视图', fontsize=11, fontweight='bold')

# M1完整轨迹
m1_full_t = np.linspace(0, np.linalg.norm(M1_0-O)/v_m, 100)
m1_full_pts = np.array([M1_0 + v_M1 * t for t in m1_full_t])
ax_global.plot(m1_full_pts[:, 0], m1_full_pts[:, 1], 'r-', linewidth=1.2,
               alpha=0.5, label='M1轨迹')
# 遮蔽窗口内的M1段加粗
t_win_end = max(Td_f) + T_life
m1_win_t = np.linspace(0, t_win_end, 30)
m1_win_pts = np.array([M1_0 + v_M1 * t for t in m1_win_t])
ax_global.plot(m1_win_pts[:, 0], m1_win_pts[:, 1], 'r-', linewidth=3,
               alpha=0.8, label=f'遮蔽窗口 (0~{t_win_end:.0f}s)')

# 无人机概略位置（用矩形框标出放大区域）
drone_xs = [p[0] for p in [FY1_0] + P_f + C_f]
drone_ys = [p[1] for p in [FY1_0] + P_f + C_f]
x_min_g, x_max_g = min(drone_xs) - 800, max(drone_xs) + 400
y_min_g, y_max_g = min(drone_ys) - 200, max(drone_ys) + 200
rect = plt.Rectangle((x_min_g, y_min_g), x_max_g - x_min_g, y_max_g - y_min_g,
                      fill=True, facecolor='yellow', alpha=0.15, edgecolor='orange',
                      linewidth=1.5, linestyle='--')
ax_global.add_patch(rect)
ax_global.annotate('放大区域', (x_max_g, y_max_g), fontsize=9,
                    color='orange', ha='right', va='bottom')

# 目标
ax_global.scatter([0], [0], c='gray', s=150, marker='s', zorder=10,
                  edgecolors='black', linewidth=1.5, label='假目标(原点)')
ax_global.scatter([0], [200], c='green', s=150, marker='s', zorder=10,
                  edgecolors='darkgreen', linewidth=1.5, label='真目标')
# 无人机起点
ax_global.scatter([FY1_0[0]], [FY1_0[1]], c='blue', s=80, marker='s', zorder=10)

ax_global.set_xlabel('x (m)', fontsize=9)
ax_global.set_ylabel('y (m)', fontsize=9)
ax_global.legend(loc='upper left', fontsize=7, framealpha=0.8)
ax_global.grid(True, alpha=0.3)
ax_global.set_aspect('equal')

# 面板B: 放大视图（左侧主图）
ax_zoom = fig2.add_axes([0.05, 0.10, 0.54, 0.82])
ax_zoom.set_title('FY1飞行路径与烟幕布设 (放大)', fontsize=13, fontweight='bold')

# 无人机路径
path = [FY1_0.copy()]
v1v = np.array([v1_f * np.cos(th1_f), v1_f * np.sin(th1_f), 0])
path.append(path[-1] + v1v * t1_f)
v2v = np.array([v2_f * np.cos(th2_f), v2_f * np.sin(th2_f), 0])
path.append(path[-1] + v2v * g12_f)
v3v = np.array([v3_f * np.cos(th3_f), v3_f * np.sin(th3_f), 0])
path.append(path[-1] + v3v * g23_f)

px = [p[0] for p in path]
py = [p[1] for p in path]

# 路径线段 + 方向箭头
ax_zoom.plot(px, py, 'k-', linewidth=2.5, alpha=0.85, label='FY1飞行路径', zorder=5)
for i in range(3):
    mid_x = (px[i] + px[i+1]) / 2
    mid_y = (py[i] + py[i+1]) / 2
    dx = px[i+1] - px[i]
    dy = py[i+1] - py[i]
    ax_zoom.annotate('', xy=(px[i+1], py[i+1]), xytext=(px[i], py[i]),
                      arrowprops=dict(arrowstyle='->', color='gray',
                                      lw=1.5, alpha=0.6))

ax_zoom.scatter(px[0], py[0], c='blue', s=150, marker='s',
                zorder=10, edgecolors='darkblue', linewidth=1.5, label='FY1初始')
for i in range(3):
    ax_zoom.scatter(px[i+1], py[i+1], c='red', s=120, marker='^',
                    zorder=10, edgecolors='darkred', linewidth=1.2)
    ax_zoom.annotate(f'  投弹{i+1}\n  t={[t1_f, t1_f+g12_f, t1_f+g12_f+g23_f][i]:.1f}s',
                      (px[i+1], py[i+1]), fontsize=8, color='darkred', va='center')

# 烟幕起爆位置 + 半径
for i in range(3):
    ax_zoom.scatter(C_f[i][0], C_f[i][1], c=colors[i], s=300, marker='*',
                    zorder=12, edgecolors='black', linewidth=1.5,
                    label=f'弹{i+1}起爆 t={Td_f[i]:.1f}s')
    ax_zoom.add_patch(plt.Circle((C_f[i][0], C_f[i][1]), smoke_r,
                                  fill=True, facecolor=colors[i], alpha=0.15,
                                  edgecolor=colors[i], linewidth=1.5, linestyle='-'))
    ax_zoom.annotate(f'r=10m', (C_f[i][0] + 12, C_f[i][1] + 5),
                      fontsize=7, color=colors[i])

# M1轨迹局部
m1_local_t = np.linspace(0, max(Td_f) + T_life, 60)
m1_local_pts = np.array([M1_0 + v_M1 * t for t in m1_local_t])
ax_zoom.plot(m1_local_pts[:, 0], m1_local_pts[:, 1], 'r-', linewidth=2,
             alpha=0.55, label='M1轨迹 (遮蔽窗口内)', zorder=3)
# M1在起爆时刻的标记（统一红色，避免和弹配色混淆）
for i in range(3):
    m1_td = M1_0 + v_M1 * Td_f[i]
    ax_zoom.scatter([m1_td[0]], [m1_td[1]], c='darkred', s=80,
                    marker='o', zorder=11, edgecolors='black', linewidth=1)
    ax_zoom.annotate(f'M1\nt={Td_f[i]:.1f}s', (m1_td[0], m1_td[1]),
                      textcoords="offset points", xytext=(0, -20),
                      fontsize=7, color='darkred', ha='center')

# 真目标方向指示（太远不在视野内）
ax_zoom.annotate('真目标 →\n(0,200) 约16km', xy=(0.98, 0.02), xycoords='axes fraction',
                  fontsize=9, color='darkgreen', ha='right', va='bottom',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85,
                            edgecolor='darkgreen'))

# 缩放：包含所有起爆位置（弹3起爆点比投弹点更靠左）
all_x = px + [C_f[i][0] for i in range(3)]
all_y = py + [C_f[i][1] for i in range(3)]
x_margin = 600
ax_zoom.set_xlim(min(all_x) - x_margin, max(all_x) + 300)
ax_zoom.set_ylim(min(all_y) - 40, max(all_y) + 80)

ax_zoom.set_xlabel('x (m)', fontsize=11)
ax_zoom.set_ylabel('y (m)', fontsize=11)
ax_zoom.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax_zoom.grid(True, alpha=0.3, linestyle='--')

fig2.suptitle('问题3: FY1轨迹与三弹烟幕布设 (xy俯视图)', fontsize=14, fontweight='bold', y=0.97)
fig2.savefig('problem3_trajectory.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"  轨迹图: problem3_trajectory.png")

# --- 图3: 摘要 ---
fig3, ax3 = plt.subplots(figsize=(10, 7))
ax3.axis('off')

# 弹1基准
r_p2_check = sim_one(FY1_0, P2_THETA, P2_V, P2_T_DROP, P2_DT)
dur_p2_cyl, _ = shielding_union([(r_p2_check[0], P2_T_DROP + P2_DT)], cyl_pts)

summary = f"""
问题3 优化摘要
{'='*48}

问题2基准 (单弹):     {dur_p2_cyl:.4f}s
3弹联合优化结果:       {dur_final:.4f}s
提升:                  +{dur_final-dur_p2_cyl:.4f}s ({(dur_final/dur_p2_cyl-1)*100:.1f}%)

最优参数:
  弹1: θ={np.degrees(th1_f):.3f}°, v={v1_f:.2f}, t_drop={t1_f:.4f}, Δt={dt1_f:.4f}
       起爆({C_f[0][0]:.0f},{C_f[0][1]:.0f},{C_f[0][2]:.0f}) @ {Td_f[0]:.3f}s
  弹2: θ={np.degrees(th2_f):.3f}°, v={v2_f:.2f}, gap={g12_f:.4f}, Δt={dt2_f:.4f}
       起爆({C_f[1][0]:.0f},{C_f[1][1]:.0f},{C_f[1][2]:.0f}) @ {Td_f[1]:.3f}s
  弹3: θ={np.degrees(th3_f):.3f}°, v={v3_f:.2f}, gap={g23_f:.4f}, Δt={dt3_f:.4f}
       起爆({C_f[2][0]:.0f},{C_f[2][1]:.0f},{C_f[2][2]:.0f}) @ {Td_f[2]:.3f}s

遮蔽区间:
"""
for i, (ts, te) in enumerate(intv_final, 1):
    summary += f"  [{ts:.4f}, {te:.4f}]  ({te-ts:.4f}s)\n"

summary += f"""
总时长: {dur_final:.4f}s
总耗时: {time.time()-t_samp:.1f}s
"""

ax3.text(0.05, 0.95, summary, transform=ax3.transAxes, fontsize=10,
         va='top', fontproperties=chinese_font,
         bbox=dict(boxstyle='round', facecolor='#F8F9F9', alpha=0.95,
                   edgecolor='#BDC3C7'))
fig3.tight_layout()
fig3.savefig('problem3_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"  摘要图: problem3_summary.png")

print(f"\n{'='*65}")
print(f"全部完成! 总耗时 {time.time()-t_samp:.1f}s")
print(f"{'='*65}")
