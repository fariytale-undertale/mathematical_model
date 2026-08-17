"""
2025 国赛 A题 问题2：单弹单目标最优投放策略
============================================
向量化加速版：批量计算替代 Python 循环
两阶段: (1) 随机采样暖启动 + DE 全局搜索 (点近似)
        (2) 圆柱采样验证 + 局部精修
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
warnings.filterwarnings('ignore', message='.*Glyph.*missing.*')

# ========== 中文字体设置 ==========
font_path = r'C:\Windows\Fonts\msyh.ttc'   # 微软雅黑

try:
    chinese_font = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [chinese_font.get_name(), 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['font.monospace'] = [chinese_font.get_name(), 'SimHei', 'DejaVu Sans Mono']
except:
    # 如果找不到文件，尝试用字体名称
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['font.monospace'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans Mono']
    
plt.rcParams['axes.unicode_minus'] = False

rcParams.update({
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 150,
})

# ============================================================
# 0. 常数
# ============================================================
g = 9.8
smoke_r, v_sink, T_life = 10.0, 3.0, 20.0
O = np.array([0.0, 0.0, 0.0])
T_pt = np.array([0.0, 200.0, 5.0])
T_cy, T_cz, T_r, T_h = 200.0, 0.0, 7.0, 10.0

M1_0 = np.array([20000.0, 0.0, 2000.0])
v_m = 300.0
v_M1 = (O - M1_0) / np.linalg.norm(O - M1_0) * v_m
FY1_0 = np.array([17800.0, 0.0, 1800.0])

# ============================================================
# 1. 向量化遮蔽时长计算（核心提速）
# ============================================================

def shielding_duration_vec(C_det, t_det, target_pts, dt=0.01):
    """
    向量化版本：对一组目标点，返回全部被遮蔽的总时长。
    target_pts: (n_pts, 3)
    """
    t_arr = np.arange(t_det, t_det + T_life, dt)
    n_t = len(t_arr)
    if n_t == 0:
        return 0.0

    # 导弹位置 (n_t, 3)
    M1_arr = M1_0 + v_M1 * t_arr[:, None]

    # 烟幕球心 (n_t, 3)
    C_arr = np.tile(C_det, (n_t, 1))
    C_arr[:, 2] -= v_sink * (t_arr - t_det)

    # 对每个目标点，找被遮蔽的时间步
    # all_blocked[i] = True 当所有目标点在第 i 步都被遮蔽
    all_blocked = np.ones(n_t, dtype=bool)

    for pt in target_pts:
        T_arr = np.tile(pt, (n_t, 1))  # (n_t, 3)
        MC = C_arr - M1_arr
        v = T_arr - M1_arr
        v2 = np.sum(v * v, axis=1)
        v2[v2 < 1e-12] = 1e-12

        s = np.sum(MC * v, axis=1) / v2
        s = np.clip(s, 0.0, 1.0)

        closest = M1_arr + s[:, None] * v
        d = np.linalg.norm(C_arr - closest, axis=1)

        all_blocked &= (d <= smoke_r)

    # 统计连续 True 区间的总时长
    edges = np.diff(np.concatenate([[False], all_blocked, [False]]).astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return np.sum((ends - starts) * dt)


def simulate(theta, v_fy1, t_drop, dt_fuze):
    v_FY1 = np.array([v_fy1 * np.cos(theta), v_fy1 * np.sin(theta), 0.0])
    FY1_drop = FY1_0 + v_FY1 * t_drop
    bomb_horiz = FY1_drop + v_FY1 * dt_fuze
    bomb_z = FY1_drop[2] - 0.5 * g * dt_fuze**2
    if bomb_z <= 0:
        return None
    return np.array([bomb_horiz[0], bomb_horiz[1], bomb_z]), t_drop + dt_fuze


# ============================================================
# 2. 圆柱采样点（预生成，复用）
# ============================================================

def build_samples():
    pts = []
    for i in range(16):
        th = 2 * np.pi * i / 16
        if np.sin(th) >= 0:
            continue
        for j in range(5):
            z = T_h * j / 4
            pts.append([T_r * np.cos(th), T_cy + T_r * np.sin(th), z])
    for z_val in [0.0, T_h]:
        for ir in [1, 2, 3]:
            rho = T_r * ir / 4
            for j in range(12):
                th = 2 * np.pi * j / 12
                if np.sin(th) >= 0:
                    continue
                pts.append([rho * np.cos(th), T_cy + rho * np.sin(th), z_val])
    return np.array(pts)


cyl_pts = build_samples()  # (n_pts, 3)
print(f"圆柱采样点: {len(cyl_pts)}")

# ============================================================
# 3. 暖启动 + DE + 精修
# ============================================================

print("=" * 65)
print("问题2：单弹单目标最优投放策略 (向量化)")
print("=" * 65)

bounds = [
    (0, np.pi *2),
    (70.0, 140.0),
    (0.0, 12.0),
    (0.5, 14.0),
]

print(f"暖启动: 随机采样 5000 点...")
np.random.seed(42)
t0 = time.time()

n_warmup = 5000
feasible = []
for _ in range(n_warmup):
    x = [np.random.uniform(*b) for b in bounds]
    r = simulate(*x)
    if r is None:
        continue
    dur = shielding_duration_vec(*r, [T_pt])
    if dur > 0:
        feasible.append((dur, np.array(x)))

feasible.sort(key=lambda v: -v[0])
feasible = feasible[:30]

n_f = len(feasible)
print(f"  找到 {n_f} 个可行解")
if n_f > 0:
    print(f"  最优: {feasible[0][0]:.4f}s")
    print(f"  前5: {[f'{d:.3f}' for d,_ in feasible[:5]]}")
print(f"  耗时: {time.time()-t0:.1f}s")

# 紧化边界
if n_f > 0:
    xs = np.array([v[1] for v in feasible])
    bounds_de = [
        (max(bounds[0][0], xs[:, 0].min() - 0.1),
         min(bounds[0][1], xs[:, 0].max() + 0.1)),
        (max(70, xs[:, 1].min() - 5), min(140, xs[:, 1].max() + 5)),
        (max(0, xs[:, 2].min() - 1), min(55, xs[:, 2].max() + 1)),
        (max(0.5, xs[:, 3].min() - 0.5), min(18, xs[:, 3].max() + 0.5)),
    ]
else:
    bounds_de = bounds

# ---- DE ----
popsize = 20
init = []
if n_f >= popsize:
    for i in range(popsize):
        x = feasible[i % n_f][1] + np.random.normal(0, 0.03, 4) * [0.3, 3, 0.3, 0.3]
        init.append(np.array([np.clip(x[j], *bounds_de[j]) for j in range(4)]))
else:
    for i in range(min(n_f, popsize)):
        x = feasible[i][1] + np.random.normal(0, 0.03, 4) * [0.3, 3, 0.3, 0.3]
        init.append(np.array([np.clip(x[j], *bounds_de[j]) for j in range(4)]))
    while len(init) < popsize:
        init.append([np.random.uniform(*b) for b in bounds_de])


def obj_point(x):
    theta, v, td, dtf = x
    r = simulate(theta, v, td, dtf)
    return 0.0 if r is None else -shielding_duration_vec(*r, [T_pt])


print(f"\nDE 全局搜索...")
t_de0 = time.time()

history = {'gen': [], 'best': []}

def track_callback(xk, convergence):
    history['gen'].append(len(history['gen']))
    history['best'].append(-obj_point(xk))

result = differential_evolution(
    obj_point, bounds_de, strategy='best1bin', maxiter=200,
    popsize=popsize, tol=0.001, mutation=(0.5, 1.5),
    recombination=0.7, seed=42, init=np.array(init), polish=False,
    callback=track_callback)
theta_d, v_d, td_d, dtf_d = result.x
dur_p = -result.fun
print(f"  耗时: {time.time()-t_de0:.1f}s, {result.nit}代/{result.nfev}次")
print(f"  θ={np.degrees(theta_d):.2f}°, v={v_d:.1f}, "
      f"t_drop={td_d:.3f}, Δt={dtf_d:.3f}, 时长={dur_p:.3f}s")

# ---- 圆柱验证 ----
r_opt = simulate(theta_d, v_d, td_d, dtf_d)
assert r_opt is not None
dur_cyl = shielding_duration_vec(*r_opt, cyl_pts)
print(f"\n  圆柱验证: {dur_cyl:.4f}s (差值 {abs(dur_cyl-dur_p)*1000:.0f}ms)")

# ---- 局部精修 ----

def obj_cyl(x):
    theta, v, td, dtf = x
    r = simulate(theta, v, td, dtf)
    return 0.0 if r is None else -shielding_duration_vec(*r, cyl_pts)


print(f"  局部精修...")
best_x, best_d = result.x.copy(), dur_cyl
# 小网格
for s in np.linspace(0.97, 1.03, 5):
    for j in range(4):
        x = result.x.copy()
        x[j] *= s
        x[j] = np.clip(x[j], *bounds_de[j])
        d = -obj_cyl(x)
        if d > best_d:
            best_d, best_x = d, x.copy()

# Nelder-Mead
try:
    nm = minimize(obj_cyl, best_x, method='Nelder-Mead',
                  bounds=bounds_de,
                  options={'xatol': 1e-4, 'fatol': 1e-5, 'maxiter': 300})
    if -nm.fun > best_d:
        best_d, best_x = -nm.fun, nm.x
except Exception as e:
    print(f"    NM: {e}")

theta_f, v_f, td_f, dtf_f = best_x
r_f = simulate(theta_f, v_f, td_f, dtf_f)
C_f, tdet_f = r_f

# ============================================================
# 4. 最终结果
# ============================================================
print(f"\n{'='*65}")
print("最终结果")
print(f"{'='*65}")
print(f"  航向角 θ    = {np.degrees(theta_f):.3f}°")
print(f"  飞行速度 v   = {v_f:.2f} m/s")
print(f"  投放时刻     = {td_f:.4f} s")
print(f"  起爆延迟     = {dtf_f:.4f} s")
print(f"  起爆时刻     = {tdet_f:.4f} s")

fd = FY1_0 + np.array([v_f*np.cos(theta_f), v_f*np.sin(theta_f), 0])*td_f
print(f"  投放位置: ({fd[0]:.1f}, {fd[1]:.1f}, {fd[2]:.1f})")
print(f"  起爆位置: ({C_f[0]:.1f}, {C_f[1]:.1f}, {C_f[2]:.1f})")

# 精确遮蔽区间
dt = 0.002
t_arr = np.arange(tdet_f, tdet_f + T_life, dt)
n_t = len(t_arr)
M1_arr = M1_0 + v_M1 * t_arr[:, None]
C_arr = np.tile(C_f, (n_t, 1))
C_arr[:, 2] -= v_sink * (t_arr - tdet_f)

all_ok = np.ones(n_t, dtype=bool)
for pt in cyl_pts:
    T_a = np.tile(pt, (n_t, 1))
    MC = C_arr - M1_arr; vv = T_a - M1_arr
    v2 = np.sum(vv*vv, axis=1); v2[v2 < 1e-12] = 1e-12
    s = np.clip(np.sum(MC*vv, axis=1)/v2, 0, 1)
    d = np.linalg.norm(C_arr - (M1_arr + s[:,None]*vv), axis=1)
    all_ok &= (d <= smoke_r)

edges = np.diff(np.concatenate([[False], all_ok, [False]]).astype(int))
st = np.where(edges == 1)[0]; en = np.where(edges == -1)[0]
print(f"\n  遮蔽区间:")
total = 0.0
for si, ei in zip(st, en):
    ts, te = tdet_f + si*dt, tdet_f + ei*dt
    dur = te - ts; total += dur
    print(f"    [{ts:.3f}, {te:.3f}]  ({dur:.3f}s)")

print(f"\n  ★ 有效遮蔽总时长: {total:.4f} s")
print(f"  与问题1比较: {total-1.435:+.4f}s ({(total/1.435-1)*100:+.1f}%)")
print(f"  总耗时: {time.time()-t0:.1f}s")

# ============================================================
# 5. 迭代过程可视化
# ============================================================

# ---- 参数敏感度图 ----
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
fig2.suptitle('参数对遮蔽时长的影响 (DE 种群末代快照)', fontsize=14, fontweight='bold')

# 在最优解附近扰动，计算遮蔽时长变化
param_names = ['航向角 θ (°)', '飞行速度 v (m/s)', '投放时刻 t_drop (s)', '起爆延迟 Δt (s)']
param_best = [np.degrees(theta_f), v_f, td_f, dtf_f]
perturb_scales = [
    np.linspace(-5, 5, 41),      # θ ±5°
    np.linspace(-15, 15, 41),    # v ±15 m/s
    np.linspace(-1.5, 1.5, 41),  # t_drop ±1.5s
    np.linspace(-1.5, 1.5, 41),  # Δt ±1.5s
]

for idx, (ax_s, name, best_val, scales) in enumerate(
    zip(axes2.flat, param_names, param_best, perturb_scales)):
    dur_vals = []
    x_vals = best_val + scales
    for delta in scales:
        x_try = best_x.copy()
        if idx == 0:
            x_try[0] = np.radians(best_val + delta)
        else:
            x_try[idx] = best_val + delta
        x_try = np.array([np.clip(x_try[j], *bounds_de[j]) for j in range(4)])
        dur_vals.append(-obj_point(x_try))

    dur_vals = np.array(dur_vals)
    ax_s.plot(x_vals, dur_vals, 'b-', linewidth=2, alpha=0.8)
    ax_s.axvline(x=best_val, color='red', linestyle='--', linewidth=1.2, alpha=0.6)
    ax_s.axhline(y=dur_p, color='green', linestyle=':', linewidth=1, alpha=0.5)

    # 标注最优
    ax_s.scatter([best_val], [dur_p], c='red', s=80, marker='*',
                 zorder=5, edgecolors='darkred')

    ax_s.set_xlabel(name, fontsize=10)
    ax_s.set_ylabel('遮蔽时长 (s)', fontsize=10)
    ax_s.set_title(f'{name} 敏感度', fontsize=12)
    ax_s.grid(True, alpha=0.3)

    # 标注变化幅度
    dur_range = dur_vals.max() - dur_vals.min()
    ax_s.text(0.98, 0.02, f'波动: {dur_range:.3f}s',
              transform=ax_s.transAxes, fontsize=8, ha='right', va='bottom',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

fig2.tight_layout()
fig2.savefig('problem2_sensitivity.png', dpi=300, bbox_inches='tight',
             facecolor='white')
print(f"\n  敏感度图已保存: problem2_sensitivity.png")

# ---- 收敛曲线 + 参数进化 (2x2) ----
gens = np.array(history['gen'])
best_vals = np.array(history['best'])

fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
fig3.suptitle('问题2 优化迭代过程全景', fontsize=15, fontweight='bold')

# (0,0) 收敛曲线
ax_c = axes3[0, 0]
ax_c.plot(gens, best_vals, 'b-', linewidth=2)
ax_c.axhline(y=total, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
if n_f > 0:
    ax_c.axhline(y=feasible[0][0], color='orange', linestyle=':', linewidth=1.2, alpha=0.7)
ax_c.fill_between(gens, 0, best_vals, alpha=0.15, color='blue')
ax_c.set_xlabel('代数', fontsize=10)
ax_c.set_ylabel('遮蔽时长 (s)', fontsize=10)
ax_c.set_title(f'收敛曲线 (初代 {best_vals[0]:.2f}s → 终代 {best_vals[-1]:.2f}s)', fontsize=11)
ax_c.legend(['DE最优', f'最终 {total:.2f}s'] +
            ([f'暖启动 {feasible[0][0]:.2f}s'] if n_f > 0 else []),
            fontsize=8)
ax_c.grid(True, alpha=0.3)

# (0,1) 暖启动散点分布
ax_s1 = axes3[0, 1]
if n_f > 0:
    ws_x = np.array([v[1] for v in feasible])
    scatter = ax_s1.scatter(np.degrees(ws_x[:, 0]), ws_x[:, 1],
                            c=[v[0] for v in feasible], cmap='YlOrRd',
                            s=40, edgecolors='gray', linewidth=0.3, alpha=0.8)
    ax_s1.scatter([np.degrees(theta_f)], [v_f], c='red', s=150, marker='*',
                  edgecolors='darkred', linewidth=1.5, zorder=10)
    cbar = plt.colorbar(scatter, ax=ax_s1)
    cbar.set_label('遮蔽时长 (s)', fontsize=8)
    ax_s1.set_xlabel('航向角 θ (°)', fontsize=10)
    ax_s1.set_ylabel('速度 v (m/s)', fontsize=10)
    ax_s1.set_title(f'暖启动可行解分布 (θ-v投影, {n_f}个)', fontsize=11)
    ax_s1.grid(True, alpha=0.3)

# (1,0) 起爆位置演化
ax_s2 = axes3[1, 0]
# 在最优解附近随机扰动，观察起爆位置和时长的关系
np.random.seed(123)
test_points = []
for _ in range(500):
    x_t = best_x + np.random.normal(0, 0.05, 4) * [0.3, 3, 0.3, 0.3]
    x_t = np.array([np.clip(x_t[j], *bounds_de[j]) for j in range(4)])
    r_t = simulate(*x_t)
    if r_t is None:
        continue
    d_t = shielding_duration_vec(*r_t, [T_pt])
    if d_t > 0:
        test_points.append((r_t[0], d_t))

if test_points:
    det_x = np.array([p[0][0] for p in test_points])
    det_y = np.array([p[0][1] for p in test_points])
    det_dur = np.array([p[1] for p in test_points])
    sc2 = ax_s2.scatter(det_x, det_y, c=det_dur, cmap='YlOrRd',
                        s=30, alpha=0.6, edgecolors='none')
    # 标记最优
    ax_s2.scatter([C_f[0]], [C_f[1]], c='red', s=200, marker='*',
                  edgecolors='darkred', linewidth=1.5, zorder=10)
    plt.colorbar(sc2, ax=ax_s2).set_label('遮蔽时长 (s)', fontsize=8)
    ax_s2.set_xlabel('起爆 x (m)', fontsize=10)
    ax_s2.set_ylabel('起爆 y (m)', fontsize=10)
    ax_s2.set_title('起爆位置 (xy) 与遮蔽时长关系', fontsize=11)
    ax_s2.grid(True, alpha=0.3)

# (1,1) 进化摘要
ax_s3 = axes3[1, 1]
ax_s3.axis('off')
summary_text = f"""
DE 优化摘要
{'─'*35}

暖启动:   {n_f} 可行解 / 5000 样本
         最优 {feasible[0][0]:.2f}s (随机采样)

DE 搜索:  {len(history['gen'])} 代, {result.nfev} 次求值
         种群 {popsize} 个, 策略 best1bin
         最优 {dur_p:.3f}s (点近似)

精修:     圆柱{len(cyl_pts)}点验证
         局部网格 + Nelder-Mead

终值:     {total:.4f}s
         比问题1提升 {(total/1.435-1)*100:.1f}%

最优参数:
  θ = {np.degrees(theta_f):.3f}°
  v = {v_f:.2f} m/s
  t_drop = {td_f:.4f} s
  Δt = {dtf_f:.4f} s
"""
ax_s3.text(0.05, 0.95, summary_text, transform=ax_s3.transAxes,
           fontsize=9, va='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#F8F9F9', alpha=0.9,
                     edgecolor='#BDC3C7'))

fig3.tight_layout()
fig3.savefig('problem2_optimization.png', dpi=300, bbox_inches='tight',
             facecolor='white')
print(f"  迭代全景图已保存: problem2_optimization.png")
