"""
2025 国赛 A题 问题4：3架无人机各投1弹最优协同策略
=====================================================
FY1、FY2、FY3 各投放1枚烟幕干扰弹，协同干扰 M1 导弹。

优化策略：12维联合 DE + 圆柱精修
  阶段1: 各无人机独立评估 (问题2框架) — 了解单机能力上限
  阶段2: 混合暖启动 (物理引导 + 随机探索)
  阶段3: DE 全局搜索 (点近似)
  阶段4: 圆柱验证 + NM 局部精修
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
M1_0 = np.array([20000.0, 0.0, 2000.0])
v_m  = 300.0
v_M1 = (O - M1_0) / np.linalg.norm(O - M1_0) * v_m

# 三架无人机初始位置
FY1_0 = np.array([17800.0, 0.0, 1800.0])
FY2_0 = np.array([12000.0, 1400.0, 1400.0])
FY3_0 = np.array([6000.0, -3000.0, 700.0])
ALL_FY = [FY1_0, FY2_0, FY3_0]
FY_NAMES = ['FY1', 'FY2', 'FY3']
FY_COLORS = ['#2196F3', '#FF9800', '#4CAF50']

# 各无人机最大自由落体时间 (z=0 触地)
MAX_DT = [np.sqrt(2 * f[2] / g) * 0.98 for f in ALL_FY]  # 留2%余量
print(f"各机最大Δt: FY1<{MAX_DT[0]:.1f}s, FY2<{MAX_DT[1]:.1f}s, FY3<{MAX_DT[2]:.1f}s")

# 问题2最优 (FY1单弹, 参考)
P2_THETA  = np.radians(7.394)
P2_V      = 98.51
P2_T_DROP = 0.0146
P2_DT     = 0.8811

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
print(f"M1飞行时间到原点: {np.linalg.norm(M1_0-O)/v_m:.1f}s")
print(f"v_M1 = ({v_M1[0]:.1f}, {v_M1[1]:.1f}, {v_M1[2]:.1f})")

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


def simulate_three_drones(x):
    """
    12维 → 3架无人机各投1弹。
    x = [θ₁,v₁,t_drop1,Δt₁, θ₂,v₂,t_drop2,Δt₂, θ₃,v₃,t_drop3,Δt₃]
    返回: (grenades_list, drop_positions, det_positions, det_times) 或 None
      其中 grenades_list = [(C_det, t_det), ...]
    """
    results = []
    for i in range(3):
        th, v, td, dt = x[i*4], x[i*4+1], x[i*4+2], x[i*4+3]
        r = sim_one(ALL_FY[i], th, v, td, dt)
        if r is None:
            return None
        results.append(r)

    grenades = [(r[0], x[i*4+2] + x[i*4+3]) for i, r in enumerate(results)]
    drops = [r[1] for r in results]
    dets = [r[0] for r in results]
    det_times = [x[i*4+2] + x[i*4+3] for i in range(3)]

    return grenades, drops, dets, det_times


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


def objective(x, target_pts):
    """12维目标函数 (返回负时长供DE最小化)"""
    r = simulate_three_drones(x)
    if r is None:
        return 0.0
    dur, _ = shielding_union(r[0], target_pts)
    return -dur


# ============================================================
# 3. 阶段1: 各无人机独立评估
# ============================================================
print(f"\n{'='*65}")
print(f"阶段1: 各无人机独立最优能力评估")
print(f"{'='*65}")

def solo_duration(fy_idx, theta, v, t_drop, dt_fuze):
    """单机单弹遮蔽时长"""
    r = sim_one(ALL_FY[fy_idx], theta, v, t_drop, dt_fuze)
    if r is None:
        return 0.0
    dur, _ = shielding_union([(r[0], t_drop + dt_fuze)], [T_pt])
    return dur


def optimize_solo(fy_idx, label, bounds_override=None):
    """对单架无人机做问题2式优化"""
    fy0 = ALL_FY[fy_idx]
    if bounds_override is None:
        bnd = [(0, 2*np.pi), (70, 140), (0, 15), (0.5, min(14, MAX_DT[fy_idx]))]
    else:
        bnd = bounds_override

    # 随机采样
    np.random.seed(42 + fy_idx * 100)
    n_samp = 3000
    best_x, best_d = None, 0.0
    feasible = []
    for _ in range(n_samp):
        x = [np.random.uniform(*b) for b in bnd]
        d = solo_duration(fy_idx, *x)
        if d > 0:
            feasible.append((d, np.array(x)))
    feasible.sort(key=lambda v: -v[0])

    if not feasible:
        print(f"  {label}: 未找到可行解!")
        return None, 0.0

    # DE
    init_pop = []
    for i in range(min(15, len(feasible))):
        xp = feasible[i][1] + np.random.normal(0, 0.03, 4) * [0.15, 3, 0.3, 0.3]
        init_pop.append(np.clip(xp, [b[0] for b in bnd], [b[1] for b in bnd]))
    while len(init_pop) < 25:
        init_pop.append([np.random.uniform(*b) for b in bnd])

    def obj_solo(x):
        return -solo_duration(fy_idx, *x)

    result = differential_evolution(
        obj_solo, bnd, strategy='best1bin', maxiter=150, popsize=25,
        tol=0.001, mutation=(0.5, 1.5), recombination=0.7,
        seed=42 + fy_idx, init=np.array(init_pop), polish=False
    )

    x_opt = result.x
    d_opt = -result.fun

    # 圆柱验证
    r_opt = sim_one(fy0, x_opt[0], x_opt[1], x_opt[2], x_opt[3])
    if r_opt:
        d_cyl, intv = shielding_union([(r_opt[0], x_opt[2] + x_opt[3])], cyl_pts)
    else:
        d_cyl = 0.0
        intv = []

    print(f"  {label}: 点近似={d_opt:.4f}s, 圆柱={d_cyl:.4f}s")
    if intv:
        print(f"         区间: {[(f'{a:.3f}', f'{b:.3f}') for a,b in intv]}")
    print(f"         θ={np.degrees(x_opt[0]):.1f}°, v={x_opt[1]:.1f}, "
          f"t_drop={x_opt[2]:.3f}, Δt={x_opt[3]:.3f}")

    return x_opt, d_cyl

solo_results = {}
for idx in range(3):
    x_solo, d_solo = optimize_solo(idx, FY_NAMES[idx])
    solo_results[idx] = (x_solo, d_solo)

# 理论最大 (各机单独最优之和, 忽略重叠)
solo_sum = sum(solo_results[i][1] for i in range(3))
print(f"\n各机独立最优之和: {solo_sum:.4f}s (理论最大, 忽略重叠)")

# ============================================================
# 4. 阶段2: 混合暖启动
# ============================================================
print(f"\n{'='*65}")
print(f"阶段2: 混合暖启动采样 (物理引导 + 随机)")
print(f"{'='*65}")

bounds_12 = []
for i in range(3):
    bounds_12.extend([
        (0, 2*np.pi),              # θ
        (70.0, 140.0),             # v
        (0.0, 18.0),               # t_drop
        (0.5, min(14.0, MAX_DT[i])),  # Δt (受限于各机高度)
    ])

print(f"变量边界:")
for i in range(3):
    b = bounds_12[i*4:(i+1)*4]
    print(f"  {FY_NAMES[i]}: θ∈[0,360]°, v∈[{b[1][0]:.0f},{b[1][1]:.0f}] m/s, "
          f"t_drop∈[{b[2][0]:.1f},{b[2][1]:.1f}]s, Δt∈[{b[3][0]:.1f},{b[3][1]:.1f}]s")

np.random.seed(12345)
t_samp = time.time()

n_warmup = 30000
warmup_results = []

# 预计算各机的"指向导弹-目标连线"的最佳航向范围
# FY2在y=1400, 需要向y≈0-100范围飞 → θ₂∈[π, 3π/2] (180°~270°, 即左+下)
# FY3在y=-3000, 需要向y≈0-100范围飞 → θ₃∈[π/2, π] (90°~180°, 即左+上)
# FY1在y=0, 基本在弹道上 → θ₁≈问题2最优附近

for k in range(n_warmup):
    x = np.zeros(12)

    rtype = np.random.random()
    if rtype < 0.35:
        # ---- 策略A: 各机独立最优附近扰动 (物理锚定) ----
        for i in range(3):
            if solo_results[i][0] is not None:
                xs = solo_results[i][0].copy()
                # 在最优解附近加噪声
                x[i*4]   = xs[0] + np.random.normal(0, 0.2)      # θ
                x[i*4+1] = xs[1] + np.random.normal(0, 5)         # v
                x[i*4+2] = max(0, xs[2] + np.random.normal(0, 0.5))  # t_drop
                x[i*4+3] = max(0.5, xs[3] + np.random.normal(0, 0.5)) # Δt
            else:
                x[i*4]   = np.random.uniform(0, 2*np.pi)
                x[i*4+1] = np.random.uniform(70, 140)
                x[i*4+2] = np.random.uniform(0, 18)
                x[i*4+3] = np.random.uniform(0.5, MAX_DT[i])
    elif rtype < 0.65:
        # ---- 策略B: 几何引导 (各机飞向导弹-目标连线) ----
        # FY1: 在弹道附近微调, θ₁≈0±30° (朝-x方向, 因为导弹飞向原点)
        x[0]   = np.random.uniform(np.pi - 0.5, np.pi + 0.5)     # ~180° (朝-x)
        x[1]   = np.random.uniform(85, 135)
        x[2]   = np.random.uniform(0, 5)
        x[3]   = np.random.uniform(0.5, min(3, MAX_DT[0]))

        # FY2: 从y=1400飞向导弹-目标连线 (y≈0-100区域), θ₂∈[π, 3π/2]
        x[4]   = np.random.uniform(np.pi - 0.3, 1.5*np.pi)       # 180°~270°
        x[5]   = np.random.uniform(80, 140)
        x[6]   = np.random.uniform(0, 10)
        x[7]   = np.random.uniform(1, min(10, MAX_DT[1]))

        # FY3: 从y=-3000飞向导弹-目标连线, θ₃∈[π/2, π]
        x[8]   = np.random.uniform(0.5*np.pi, np.pi + 0.3)       # 90°~180°
        x[9]   = np.random.uniform(80, 140)
        x[10]  = np.random.uniform(0, 12)
        x[11]  = np.random.uniform(1, min(8, MAX_DT[2]))
    else:
        # ---- 策略C: 全空间探索 ----
        for j in range(12):
            x[j] = np.random.uniform(*bounds_12[j])

    # 裁剪到边界
    for j in range(12):
        x[j] = np.clip(x[j], bounds_12[j][0], bounds_12[j][1])

    warmup_results.append(x)

# 批量评估
print(f"  评估 {n_warmup} 个样本...")
feasible = []
for i, x in enumerate(warmup_results):
    r = simulate_three_drones(x)
    if r is None:
        continue
    dur, _ = shielding_union(r[0], [T_pt])
    if dur > 0.3:  # 过滤太短的
        feasible.append((dur, x))
    if (i+1) % 10000 == 0:
        print(f"    {i+1}/{n_warmup}... (已找到 {len(feasible)} 可行解)")

feasible.sort(key=lambda v: -v[0])
n_feas = len(feasible)
print(f"\n  有效样本: {n_feas}")
if n_feas > 0:
    print(f"  Top5: {[f'{r[0]:.3f}s' for r in feasible[:5]]}")
    best_warm = feasible[0]
    print(f"  Top1 参数: "
          f"θ₁={np.degrees(best_warm[1][0]):.0f}°, v₁={best_warm[1][1]:.0f}, "
          f"t1={best_warm[1][2]:.2f}, Δt₁={best_warm[1][3]:.2f} | "
          f"θ₂={np.degrees(best_warm[1][4]):.0f}°, v₂={best_warm[1][5]:.0f}, "
          f"t2={best_warm[1][6]:.2f}, Δt₂={best_warm[1][7]:.2f} | "
          f"θ₃={np.degrees(best_warm[1][8]):.0f}°, v₃={best_warm[1][9]:.0f}, "
          f"t3={best_warm[1][10]:.2f}, Δt₃={best_warm[1][11]:.2f}")
print(f"  耗时: {time.time()-t_samp:.1f}s")

# ============================================================
# 5. 阶段3: DE 全局搜索
# ============================================================
print(f"\n{'='*65}")
print(f"阶段3: DE全局搜索 (60个体, 400代, 点近似)")
print(f"{'='*65}")

popsize = 60
init_pop = []

# 从暖启动构造初始种群
if n_feas >= 30:
    for i in range(min(30, n_feas)):
        xp = feasible[i][1] + np.random.normal(0, 0.03, 12) * [
            0.15, 3, 0.2, 0.3, 0.3, 4, 0.5, 0.5, 0.3, 4, 0.5, 0.5]
        init_pop.append(np.clip(xp,
                                [b[0] for b in bounds_12],
                                [b[1] for b in bounds_12]))
else:
    for i in range(min(n_feas, popsize)):
        init_pop.append(feasible[i][1])

# 补充: 多样化个体
while len(init_pop) < popsize:
    x = np.zeros(12)
    rtype = np.random.random()
    if rtype < 0.5 and n_feas > 0:
        # 在暖启动最优附近扰动
        ref = feasible[np.random.randint(0, min(20, n_feas))][1]
        for j in range(12):
            scale = [0.15, 3, 0.3, 0.4, 0.3, 4, 0.6, 0.6, 0.3, 4, 0.6, 0.6][j]
            x[j] = ref[j] + np.random.normal(0, 0.05) * scale
    else:
        for j in range(12):
            x[j] = np.random.uniform(*bounds_12[j])
    x = np.clip(x, [b[0] for b in bounds_12], [b[1] for b in bounds_12])
    init_pop.append(x)

print(f"  初始种群: {len(init_pop)} 个体")
if n_feas > 0:
    init_fitness = [-objective(x, [T_pt]) for x in init_pop[:10]]
    print(f"  前10个体适应度: {[f'{f:.3f}' for f in init_fitness]}")

t_de = time.time()
de_history = {'best': [], 'mean': []}

def cb_de(xk, convergence):
    val = -objective(xk, [T_pt])
    de_history['best'].append(val)
    if len(de_history['best']) % 100 == 0:
        print(f"    代{len(de_history['best'])}: best={de_history['best'][-1]:.3f}s")

result_de = differential_evolution(
    lambda x: objective(x, [T_pt]),
    bounds_12,
    strategy='best1bin',
    maxiter=400,
    popsize=popsize,
    tol=0.00001,
    mutation=(0.5, 1.8),
    recombination=0.7,
    seed=42,
    init=np.array(init_pop),
    polish=False,
    callback=cb_de,
)

dur_de = -result_de.fun
print(f"\n  DE结果: {dur_de:.4f}s (点近似)")
print(f"  代数: {result_de.nit}, 评估: {result_de.nfev}")
print(f"  耗时: {time.time()-t_de:.1f}s")

# 圆柱验证
r_de = simulate_three_drones(result_de.x)
if r_de:
    dur_de_cyl, intv_de_cyl = shielding_union(r_de[0], cyl_pts)
else:
    dur_de_cyl = 0.0
    intv_de_cyl = []
print(f"  圆柱验证: {dur_de_cyl:.4f}s")

# ============================================================
# 6. 阶段4: 圆柱精修 (NM + 网格)
# ============================================================
print(f"\n{'='*65}")
print(f"阶段4: NM+网格圆柱精修")
print(f"{'='*65}")

x_current = result_de.x.copy()
dur_current = dur_de_cyl if dur_de_cyl > 0 else dur_de

# 6a. Nelder-Mead 圆柱精修
print(f"  NM精修 (圆柱, 从 {dur_current:.4f}s 出发)...")
try:
    nm = minimize(
        lambda x: objective(x, cyl_pts),
        x_current,
        method='Nelder-Mead',
        options={'xatol': 1e-5, 'fatol': 1e-6, 'maxiter': 1000}
    )
    dur_nm = -nm.fun
    if dur_nm > dur_current:
        print(f"  NM提升: {dur_current:.4f} → {dur_nm:.4f}s (+{dur_nm-dur_current:.4f}s)")
        x_best = nm.x.copy()
    else:
        print(f"  NM未显著提升, 保持DE结果")
        x_best = x_current.copy()
        dur_nm = dur_current
except Exception as e:
    print(f"  NM失败: {e}, 保持DE结果")
    x_best = x_current.copy()
    dur_nm = dur_current

# 6b. 网格精修
print(f"  网格精修...")
best_dur = dur_nm
for scale in np.linspace(0.96, 1.04, 7):
    for j in range(12):
        xt = x_best.copy()
        xt[j] *= scale
        xt[j] = np.clip(xt[j], bounds_12[j][0], bounds_12[j][1])
        d = -objective(xt, cyl_pts)
        if d > best_dur:
            best_dur = d
            x_best = xt.copy()
            print(f"    网格提升: {best_dur:.4f}s (参数{j}: {scale:.3f}x)")

dur_best = best_dur

# ============================================================
# 6c. 逐机微调 (各机参数局部搜一遍)
# ============================================================
print(f"\n  逐机微调...")
for i_drone in range(3):
    improved = True
    round_count = 0
    while improved and round_count < 3:
        improved = False
        round_count += 1
        for s in np.linspace(0.98, 1.02, 9):
            for j in range(i_drone*4, i_drone*4+4):
                xt = x_best.copy()
                xt[j] *= s
                xt[j] = np.clip(xt[j], bounds_12[j][0], bounds_12[j][1])
                d = -objective(xt, cyl_pts)
                if d > dur_best + 0.001:
                    dur_best = d
                    x_best = xt.copy()
                    improved = True
                    print(f"    {FY_NAMES[i_drone]}参数{j%4}: {dur_best:.4f}s")

# Another NM pass from the refined point
try:
    nm2 = minimize(
        lambda x: objective(x, cyl_pts),
        x_best,
        method='Nelder-Mead',
        options={'xatol': 1e-6, 'fatol': 1e-7, 'maxiter': 600}
    )
    dur_nm2 = -nm2.fun
    if dur_nm2 > dur_best:
        print(f"  NM2提升: {dur_best:.4f} → {dur_nm2:.4f}s")
        x_best = nm2.x.copy()
        dur_best = dur_nm2
except Exception as e:
    print(f"  NM2: {e}")

# ============================================================
# 6d. FY3 专项拯救 (宽边界 + 物理引导)
# ============================================================
r_check = simulate_three_drones(x_best)
fy3_solo, fy3_intv = shielding_union([r_check[0][2]], cyl_pts, dt=0.005)
fy3_marginal = dur_best - shielding_union(r_check[0][:2], cyl_pts)[0]

print(f"\n{'='*65}")
print(f"阶段4B: FY3贡献诊断")
print(f"  FY3单独={fy3_solo:.4f}s, 边际贡献={fy3_marginal:.4f}s")

if fy3_marginal < 0.05:
    print(f"  FY3当前几乎无贡献, 启动专项拯救...")
    print(f"{'='*65}")

    # 宽边界 + 物理引导采样
    fy3_wide_bounds = [
        (0, 2*np.pi),        # θ₃
        (70.0, 140.0),       # v₃
        (0.0, 30.0),         # t_drop₃: 放宽
        (0.5, MAX_DT[2]),    # Δt₃
    ]

    # 固定FY1+FY2最优参数
    x12_fixed = x_best[:8].copy()

    def obj_fy3_only(x4, target_pts):
        """固定FY1+FY2, 只优化FY3的4个参数"""
        grenades_12 = [(r_check[0][0][0], r_check[0][0][1]),
                        (r_check[0][1][0], r_check[0][1][1])]
        th3, v3, td3, dt3 = x4
        r3 = sim_one(FY3_0, th3, v3, td3, dt3)
        if r3 is None:
            return 0.0
        C3, P3 = r3
        td3_det = td3 + dt3
        dur, _ = shielding_union(grenades_12 + [(C3, td3_det)], target_pts)
        return -dur

    # 几何分析: FY3在(6000,-3000,700), M1→T视线在y≈0~200
    # 要飞向视线, θ₃应在[π/2, π]范围(东北方向 = +y, -x)
    np.random.seed(99999)
    fy3_feasible = []

    print(f"  物理引导采样: θ₃∈[90°,180°]飞向导弹-目标连线...")
    for _ in range(15000):
        th3 = np.random.uniform(0.4*np.pi, np.pi + 0.3)   # 72°~198°
        v3  = np.random.uniform(80, 140)
        td3 = np.random.uniform(0, 30)
        dt3 = np.random.uniform(0.5, MAX_DT[2])
        x4 = [th3, v3, td3, dt3]
        d = -obj_fy3_only(x4, [T_pt])
        if d > dur_best + 0.01:
            fy3_feasible.append((d, np.array(x4)))

    fy3_feasible.sort(key=lambda v: -v[0])
    n_fy3 = len(fy3_feasible)
    print(f"  优于当前解: {n_fy3} 个, 最优: {fy3_feasible[0][0]:.4f}s" if n_fy3 else "  无改进解")

    if n_fy3 > 0 and fy3_feasible[0][0] > dur_best + 0.02:
        # DE on FY3 only
        init_fy3 = []
        for i in range(min(20, n_fy3)):
            xp = fy3_feasible[i][1] + np.random.normal(0, 0.03, 4) * [0.2, 3, 0.5, 0.5]
            init_fy3.append(np.clip(xp,
                                    [b[0] for b in fy3_wide_bounds],
                                    [b[1] for b in fy3_wide_bounds]))
        while len(init_fy3) < 30:
            th3 = np.random.uniform(0.3*np.pi, np.pi + 0.5)
            v3 = np.random.uniform(80, 140)
            td3 = np.random.uniform(0, 30)
            dt3 = np.random.uniform(0.5, MAX_DT[2])
            init_fy3.append([th3, v3, td3, dt3])

        print(f"  FY3 DE搜索 (30个体, 200代)...")
        t_fy3de = time.time()
        result_fy3 = differential_evolution(
            lambda x: obj_fy3_only(x, [T_pt]),
            fy3_wide_bounds,
            strategy='best1bin', maxiter=200, popsize=30,
            tol=0.0001, mutation=(0.5, 1.5), recombination=0.7,
            seed=99999, init=np.array(init_fy3), polish=False
        )

        x_try = x_best.copy()
        x_try[8:] = result_fy3.x
        r_try = simulate_three_drones(x_try)
        if r_try:
            dur_try, _ = shielding_union(r_try[0], cyl_pts)
            fy3_new_solo, _ = shielding_union([r_try[0][2]], cyl_pts)
            print(f"  FY3 DE: 联合={dur_try:.4f}s, FY3单独={fy3_new_solo:.4f}s "
                  f"(之前联合={dur_best:.4f}s)")

            if dur_try > dur_best + 0.01:
                x_best = x_try.copy()
                dur_best = dur_try
                print(f"  ★ FY3拯救成功! +{dur_try-dur_best+0.01:.4f}s")
    else:
        print(f"  FY3物理上难以贡献M1遮蔽 (y偏移3000m过大)")

# ============================================================
# 6e. 降维回退: FY1+FY2 8参数优化 (当FY3无法贡献时)
# ============================================================
r_final_check = simulate_three_drones(x_best)
fy3_check_solo, _ = shielding_union([r_final_check[0][2]], cyl_pts, dt=0.005)

if fy3_check_solo < 0.05:
    print(f"\n{'='*65}")
    print(f"阶段4C: FY1+FY2降维优化 (FY3确认无贡献)")
    print(f"{'='*65}")

    bounds_8 = bounds_12[:8]
    # 用当前最优的FY1+FY2作为起点
    x8_init = x_best[:8].copy()
    dur8_init = shielding_union(r_final_check[0][:2], cyl_pts)[0]
    print(f"  当前FY1+FY2: {dur8_init:.4f}s")

    # NM 从当前解出发
    def obj_8(x8, target_pts):
        results = []
        for i in range(2):
            th, v, td, dt = x8[i*4], x8[i*4+1], x8[i*4+2], x8[i*4+3]
            r = sim_one(ALL_FY[i], th, v, td, dt)
            if r is None:
                return 0.0
            results.append(r)
        grenades = [(r[0], x8[i*4+2] + x8[i*4+3]) for i, r in enumerate(results)]
        dur, _ = shielding_union(grenades, target_pts)
        return -dur

    # NM精修
    try:
        nm8 = minimize(
            lambda x: obj_8(x, cyl_pts),
            x8_init,
            method='Nelder-Mead',
            options={'xatol': 1e-5, 'fatol': 1e-6, 'maxiter': 800}
        )
        dur_nm8 = -nm8.fun
        if dur_nm8 > dur8_init:
            x8_best = nm8.x.copy()
            print(f"  NM8提升: {dur8_init:.4f} → {dur_nm8:.4f}s (+{dur_nm8-dur8_init:.4f}s)")
        else:
            x8_best = x8_init.copy()
            dur_nm8 = dur8_init
    except Exception as e:
        print(f"  NM8: {e}")
        x8_best = x8_init.copy()
        dur_nm8 = dur8_init

    # 网格
    for scale in np.linspace(0.97, 1.03, 7):
        for j in range(8):
            xt = x8_best.copy()
            xt[j] *= scale
            xt[j] = np.clip(xt[j], bounds_8[j][0], bounds_8[j][1])
            d = -obj_8(xt, cyl_pts)
            if d > dur_nm8 + 0.001:
                dur_nm8 = d
                x8_best = xt.copy()
                print(f"    网格提升: {dur_nm8:.4f}s")

    # 逐机微调
    for _ in range(3):
        for i_d in range(2):
            for s in np.linspace(0.98, 1.02, 7):
                for j in range(i_d*4, i_d*4+4):
                    xt = x8_best.copy()
                    xt[j] *= s
                    xt[j] = np.clip(xt[j], bounds_8[j][0], bounds_8[j][1])
                    d = -obj_8(xt, cyl_pts)
                    if d > dur_nm8 + 0.0005:
                        dur_nm8 = d
                        x8_best = xt.copy()

    # 更新x_best (FY1+FY2更新, FY3保留但标记无贡献)
    x_best[:8] = x8_best
    dur_best = dur_nm8
    print(f"  FY1+FY2最终: {dur_best:.4f}s")

# ============================================================
# 7. 最终结果提取
# ============================================================
r_final = simulate_three_drones(x_best)
assert r_final is not None
grenades_f, P_f, C_f, Td_f = r_final

# 精确验证
dt_fine = 0.002
dur_final_cyl, intv_final = shielding_union(grenades_f, cyl_pts, dt=dt_fine)
dur_final_pt, intv_final_pt = shielding_union(grenades_f, [T_pt], dt=dt_fine)

print(f"\n{'='*65}")
print(f"最终结果")
print(f"{'='*65}")

for i in range(3):
    th = np.degrees(x_best[i*4])
    v  = x_best[i*4+1]
    td = x_best[i*4+2]
    dtf = x_best[i*4+3]
    print(f"\n  ┌─── {FY_NAMES[i]} ───")
    print(f"  │ 航向角 θ     = {th:.3f}°")
    print(f"  │ 飞行速度 v   = {v:.2f} m/s")
    print(f"  │ 投放时刻     = {td:.4f} s")
    print(f"  │ 起爆延迟     = {dtf:.4f} s")
    print(f"  │ 起爆时刻     = {Td_f[i]:.4f} s")
    print(f"  │ 投放位置     = ({P_f[i][0]:.1f}, {P_f[i][1]:.1f}, {P_f[i][2]:.1f})")
    print(f"  │ 起爆位置     = ({C_f[i][0]:.1f}, {C_f[i][1]:.1f}, {C_f[i][2]:.1f})")
    print(f"  └{'─'*30}")

print(f"\n  遮蔽区间 (圆柱验证, dt={dt_fine*1000:.0f}ms):")
for i, (ts, te) in enumerate(intv_final, 1):
    print(f"    [{ts:.4f}, {te:.4f}]  ({te-ts:.4f}s)")
print(f"\n  ★ 有效遮蔽总时长: {dur_final_cyl:.4f}s (圆柱验证)")
print(f"  ★ 点近似:         {dur_final_pt:.4f}s")
print(f"  总耗时: {time.time()-t_samp:.1f}s")

# 各弹贡献
print(f"\n各弹贡献分析:")
for i in range(3):
    sd, si = shielding_union([grenades_f[i]], cyl_pts, dt=dt_fine)
    others = [grenades_f[j] for j in range(3) if j != i]
    wd, _ = shielding_union(others, cyl_pts, dt=dt_fine)
    print(f"  {FY_NAMES[i]}: 单独={sd:.4f}s, 并集边际贡献={dur_final_cyl-wd:.4f}s, "
          f"区间={[(f'{a:.3f}',f'{b:.3f}') for a,b in si]}")

# 重叠分析
overlap_analysis = 0.0
for i in range(3):
    overlap_analysis += solo_results[i][1] if solo_results[i][0] is not None else 0
overlap_analysis -= dur_final_cyl
print(f"  重叠损耗: {overlap_analysis:.4f}s (各机最优之和{solo_sum:.4f} - 联合{dur_final_cyl:.4f})")

# ============================================================
# 8. 保存 result2.xlsx
# ============================================================
print(f"\n{'='*65}")
print(f"保存 result2.xlsx")
print(f"{'='*65}")

import pandas as pd

# 构建数据 — 按模板格式
data = {}
data['无人机编号'] = ['FY1', 'FY2', 'FY3', '', '注：以x轴为正向，逆时针方向为正，取值0~360（度）。']
data['无人机运动方向'] = [
    round(np.degrees(x_best[0]), 2),
    round(np.degrees(x_best[4]), 2),
    round(np.degrees(x_best[8]), 2),
    '', ''
]
data['无人机运动速度 (m/s)'] = [
    round(x_best[1], 2), round(x_best[5], 2), round(x_best[9], 2), '', ''
]
data['烟幕干扰弹投放点的x坐标 (m)'] = [
    round(P_f[0][0], 2), round(P_f[1][0], 2), round(P_f[2][0], 2), '', ''
]
data['烟幕干扰弹投放点的y坐标 (m)'] = [
    round(P_f[0][1], 2), round(P_f[1][1], 2), round(P_f[2][1], 2), '', ''
]
data['烟幕干扰弹投放点的z坐标 (m)'] = [
    round(P_f[0][2], 2), round(P_f[1][2], 2), round(P_f[2][2], 2), '', ''
]
data['烟幕干扰弹起爆点的x坐标 (m)'] = [
    round(C_f[0][0], 2), round(C_f[1][0], 2), round(C_f[2][0], 2), '', ''
]
data['烟幕干扰弹起爆点的y坐标 (m)'] = [
    round(C_f[0][1], 2), round(C_f[1][1], 2), round(C_f[2][1], 2), '', ''
]
data['烟幕干扰弹起爆点的z坐标 (m)'] = [
    round(C_f[0][2], 2), round(C_f[1][2], 2), round(C_f[2][2], 2), '', ''
]
data['有效干扰时长 (s)'] = [
    round(dur_final_cyl, 2), round(dur_final_cyl, 2), round(dur_final_cyl, 2), '', ''
]

df_out = pd.DataFrame(data)
df_out.to_excel('result2.xlsx', sheet_name='Sheet1', index=False)
print(f"  已保存 result2.xlsx")

# 确认
df_check = pd.read_excel('result2.xlsx')
print(f"\n  保存内容确认:")
print(df_check.to_string())

# ============================================================
# 9. 可视化
# ============================================================
print(f"\n{'='*65}")
print(f"生成可视化图表")
print(f"{'='*65}")

t_viz = time.time()

# --- 图1: 遮蔽区间甘特图 ---
fig1, ax1 = plt.subplots(figsize=(14, 5))
fig1.suptitle('问题4: 三机协同遮蔽区间甘特图', fontsize=14, fontweight='bold')

t_gmin = min(Td_f)
t_gmax = max(Td_f) + T_life

for i in range(3):
    sd, si = shielding_union([grenades_f[i]], cyl_pts, dt=dt_fine)
    for ts, te in si:
        ax1.barh(i, te - ts, left=ts, height=0.5, color=FY_COLORS[i],
                 alpha=0.85, edgecolor='white', linewidth=0.5)

# 并集
for ts, te in intv_final:
    ax1.barh(3, te - ts, left=ts, height=0.5, color='#E91E63',
             alpha=0.7, edgecolor='white', linewidth=0.5)

for i in range(3):
    ax1.axvline(x=Td_f[i], color=FY_COLORS[i], linestyle='--', alpha=0.4, linewidth=1)
    ax1.text(Td_f[i], i + 0.4, f'{Td_f[i]:.2f}s', fontsize=7,
             ha='center', va='bottom', color=FY_COLORS[i])

ax1.set_yticks(range(4))
ax1.set_yticklabels(['FY1', 'FY2', 'FY3', '并集'], fontsize=11)
ax1.set_xlabel('时间 (s)', fontsize=11)
ax1.set_xlim(t_gmin - 1, t_gmax + 1)
ax1.grid(True, alpha=0.3, axis='x')
fig1.tight_layout()
fig1.savefig('problem4_gantt.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"  甘特图: problem4_gantt.png")

# --- 图2: 轨迹俯视图 (全局+放大) ---
fig2 = plt.figure(figsize=(18, 8))

# 面板A: 全局视图
ax_global = fig2.add_axes([0.62, 0.15, 0.35, 0.72])
ax_global.set_title('全局视图', fontsize=11, fontweight='bold')

# M1完整轨迹
m1_full_t = np.linspace(0, np.linalg.norm(M1_0-O)/v_m, 100)
m1_full_pts = np.array([M1_0 + v_M1 * t for t in m1_full_t])
ax_global.plot(m1_full_pts[:, 0], m1_full_pts[:, 1], 'r-', linewidth=1.2,
               alpha=0.5, label='M1轨迹')
# 遮蔽窗口内的M1段加粗
t_win_end = max(Td_f) + T_life
m1_win_t = np.linspace(0, t_win_end, 50)
m1_win_pts = np.array([M1_0 + v_M1 * t for t in m1_win_t])
ax_global.plot(m1_win_pts[:, 0], m1_win_pts[:, 1], 'r-', linewidth=3,
               alpha=0.8, label=f'M1 (遮蔽窗口)')

# 目标
ax_global.scatter([0], [0], c='gray', s=150, marker='s', zorder=10,
                  edgecolors='black', linewidth=1.5, label='假目标(原点)')
ax_global.scatter([0], [200], c='green', s=150, marker='s', zorder=10,
                  edgecolors='darkgreen', linewidth=1.5, label='真目标')

# 三架无人机起始位置
for i in range(3):
    ax_global.scatter([ALL_FY[i][0]], [ALL_FY[i][1]], c=FY_COLORS[i], s=100,
                      marker='s', zorder=10, edgecolors='black', linewidth=1)
    ax_global.annotate(f'{FY_NAMES[i]}', (ALL_FY[i][0], ALL_FY[i][1]),
                       textcoords="offset points", xytext=(5, -12),
                       fontsize=8, color=FY_COLORS[i], fontweight='bold')

# 起爆位置
for i in range(3):
    ax_global.scatter([C_f[i][0]], [C_f[i][1]], c=FY_COLORS[i], s=120, marker='*',
                      zorder=12, edgecolors='black', linewidth=1)

ax_global.set_xlabel('x (m)', fontsize=9)
ax_global.set_ylabel('y (m)', fontsize=9)
ax_global.legend(loc='upper left', fontsize=7, framealpha=0.8)
ax_global.grid(True, alpha=0.3)
ax_global.set_aspect('equal')

# 面板B: 放大视图
ax_zoom = fig2.add_axes([0.05, 0.10, 0.54, 0.82])
ax_zoom.set_title('三机飞行路径与烟幕布设 (放大)', fontsize=13, fontweight='bold')

# 各机飞行路径
for i in range(3):
    fy0 = ALL_FY[i]
    th = x_best[i*4]
    v  = x_best[i*4+1]
    td = x_best[i*4+2]
    v_vec = np.array([v * np.cos(th), v * np.sin(th), 0])

    path_start = fy0.copy()
    path_drop = fy0 + v_vec * td

    px_vals = [path_start[0], path_drop[0]]
    py_vals = [path_start[1], path_drop[1]]

    ax_zoom.plot(px_vals, py_vals, '-', color=FY_COLORS[i], linewidth=2.5,
                 alpha=0.85, label=f'{FY_NAMES[i]}路径', zorder=5)
    # 箭头
    ax_zoom.annotate('', xy=(px_vals[1], py_vals[1]),
                     xytext=(px_vals[0], py_vals[0]),
                     arrowprops=dict(arrowstyle='->', color=FY_COLORS[i],
                                     lw=1.5, alpha=0.6))

    # 起点
    ax_zoom.scatter([px_vals[0]], [py_vals[0]], c=FY_COLORS[i], s=100, marker='s',
                    zorder=10, edgecolors='black', linewidth=1.2)
    # 投弹点
    ax_zoom.scatter([px_vals[1]], [py_vals[1]], c=FY_COLORS[i], s=80, marker='^',
                    zorder=10, edgecolors='black', linewidth=1)
    # 起爆位置 + 烟幕范围
    ax_zoom.scatter([C_f[i][0]], [C_f[i][1]], c=FY_COLORS[i], s=250, marker='*',
                    zorder=12, edgecolors='black', linewidth=1.5)
    ax_zoom.add_patch(plt.Circle((C_f[i][0], C_f[i][1]), smoke_r,
                                  fill=True, facecolor=FY_COLORS[i], alpha=0.12,
                                  edgecolor=FY_COLORS[i], linewidth=1.5, linestyle='-'))

    # 各弹起爆时刻标注
    ax_zoom.annotate(f'{FY_NAMES[i]}起爆\nt={Td_f[i]:.2f}s',
                     (C_f[i][0], C_f[i][1]),
                     textcoords="offset points", xytext=(8, 8),
                     fontsize=7, color=FY_COLORS[i])

# M1轨迹局部
m1_local_t = np.linspace(0, max(Td_f) + T_life, 60)
m1_local_pts = np.array([M1_0 + v_M1 * t for t in m1_local_t])
ax_zoom.plot(m1_local_pts[:, 0], m1_local_pts[:, 1], 'r-', linewidth=2,
             alpha=0.5, label='M1轨迹', zorder=3)

# M1在各起爆时刻的位置
for i in range(3):
    m1_at_det = M1_0 + v_M1 * Td_f[i]
    ax_zoom.scatter([m1_at_det[0]], [m1_at_det[1]], c='darkred', s=60,
                    marker='o', zorder=11, edgecolors='black', linewidth=0.8)

# 真目标方向
ax_zoom.annotate('真目标(0,200) →', xy=(0.98, 0.02), xycoords='axes fraction',
                 fontsize=9, color='darkgreen', ha='right', va='bottom',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85,
                           edgecolor='darkgreen'))

# 缩放范围
all_x = [p[0] for p in [FY1_0, FY2_0, FY3_0]] + [P_f[i][0] for i in range(3)] + [C_f[i][0] for i in range(3)]
all_y = [p[1] for p in [FY1_0, FY2_0, FY3_0]] + [P_f[i][1] for i in range(3)] + [C_f[i][1] for i in range(3)]
x_margin = 800
y_margin = 500
ax_zoom.set_xlim(min(all_x) - x_margin, max(all_x) + 400)
ax_zoom.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)

ax_zoom.set_xlabel('x (m)', fontsize=11)
ax_zoom.set_ylabel('y (m)', fontsize=11)
ax_zoom.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax_zoom.grid(True, alpha=0.3, linestyle='--')

fig2.suptitle('问题4: 三机飞行轨迹与烟幕布设 (xy俯视图)', fontsize=14, fontweight='bold', y=0.97)
fig2.savefig('problem4_trajectory.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"  轨迹图: problem4_trajectory.png")

# --- 图3: 收敛曲线 ---
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle('问题4 优化过程', fontsize=14, fontweight='bold')

# 收敛曲线
ax_conv = axes3[0]
gens = np.arange(len(de_history['best']))
ax_conv.plot(gens, de_history['best'], 'b-', linewidth=2, label='DE 最优')
ax_conv.fill_between(gens, 0, de_history['best'], alpha=0.1, color='blue')
if n_feas > 0:
    ax_conv.axhline(y=feasible[0][0], color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'暖启动最优 {feasible[0][0]:.2f}s')
ax_conv.axhline(y=dur_final_cyl, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'最终 {dur_final_cyl:.2f}s')
ax_conv.set_xlabel('DE 代数', fontsize=11)
ax_conv.set_ylabel('遮蔽时长 (s)', fontsize=11)
ax_conv.set_title(f'收敛曲线', fontsize=12)
ax_conv.legend(fontsize=8)
ax_conv.grid(True, alpha=0.3)

# 各弹遮蔽区间对比
ax_intv = axes3[1]
for i in range(3):
    sd, si = shielding_union([grenades_f[i]], cyl_pts, dt=dt_fine)
    for ts, te in si:
        ax_intv.barh(3-i, te - ts, left=ts, height=0.6, color=FY_COLORS[i],
                     alpha=0.8, edgecolor='white', linewidth=0.5)

# 并集区间标注
for ts, te in intv_final:
    ax_intv.axvspan(ts, te, alpha=0.15, color='red')
    ax_intv.text((ts+te)/2, 3.7, f'{te-ts:.2f}s', ha='center', fontsize=8, color='darkred')

ax_intv.set_yticks([3.3, 2.3, 1.3, 0.3])
ax_intv.set_yticklabels(['并集', 'FY3', 'FY2', 'FY1'], fontsize=10)
ax_intv.set_xlabel('时间 (s)', fontsize=11)
ax_intv.set_title(f'遮蔽区间分解 (总计 {dur_final_cyl:.3f}s)', fontsize=12)
ax_intv.grid(True, alpha=0.3, axis='x')

fig3.tight_layout()
fig3.savefig('problem4_convergence.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"  收敛图: problem4_convergence.png")

# --- 图4: 摘要 ---
fig4, ax4 = plt.subplots(figsize=(11, 8))
ax4.axis('off')

summary_text = f"""
问题4 优化摘要
{'='*55}

场景: FY1+FY2+FY3 各投1弹, 协同干扰 M1

各机独立最优:
"""
for i in range(3):
    if solo_results[i][0] is not None:
        xs = solo_results[i][0]
        sd = solo_results[i][1]
        summary_text += f"  {FY_NAMES[i]}: {sd:.3f}s  (θ={np.degrees(xs[0]):.0f}°, v={xs[1]:.0f}, t_drop={xs[2]:.2f}, Δt={xs[3]:.2f})\n"

summary_text += f"""
三机联合最优: {dur_final_cyl:.4f}s
提升 vs 最佳单机: +{dur_final_cyl - max(solo_results[i][1] for i in range(3)):.4f}s
各机独立之和: {solo_sum:.4f}s (重叠损耗 {overlap_analysis:.4f}s)

最优参数:
"""
for i in range(3):
    th = np.degrees(x_best[i*4])
    v  = x_best[i*4+1]
    td = x_best[i*4+2]
    dtf = x_best[i*4+3]
    summary_text += f"  {FY_NAMES[i]}: θ={th:.2f}°, v={v:.2f}, t_drop={td:.4f}, Δt={dtf:.4f}\n"
    summary_text += f"         起爆({C_f[i][0]:.0f},{C_f[i][1]:.0f},{C_f[i][2]:.0f}) @ {Td_f[i]:.3f}s\n"

summary_text += f"""
遮蔽区间:
"""
for i, (ts, te) in enumerate(intv_final, 1):
    summary_text += f"  [{ts:.4f}, {te:.4f}]  ({te-ts:.4f}s)\n"

summary_text += f"""
各弹贡献:
"""
for i in range(3):
    sd, si = shielding_union([grenades_f[i]], cyl_pts, dt=dt_fine)
    others = [grenades_f[j] for j in range(3) if j != i]
    wd, _ = shielding_union(others, cyl_pts, dt=dt_fine)
    summary_text += f"  {FY_NAMES[i]}: 单独={sd:.4f}s, 边际={dur_final_cyl-wd:.4f}s\n"

summary_text += f"""
总耗时: {time.time()-t_samp:.1f}s
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
         va='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#F8F9F9', alpha=0.95,
                   edgecolor='#BDC3C7'))
fig4.tight_layout()
fig4.savefig('problem4_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"  摘要图: problem4_summary.png")

print(f"\n{'='*65}")
print(f"全部完成! 总耗时 {time.time()-t_samp:.1f}s")
print(f"{'='*65}")
