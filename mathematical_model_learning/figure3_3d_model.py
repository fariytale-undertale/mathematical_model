"""
图三：烟幕遮蔽原理示意图（纯示意，非真实比例）
目的：阐述烟幕干扰弹遮蔽原理，物理量取整数便于理解
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches

# ============================================================
# 0. 中文字体
# ============================================================
plt.rcParams.update({
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ============================================================
# 1. 示意图几何参数（全部取整数，范围 ~100m）
# ============================================================
# 坐标原点 = 假目标（诱饵）
O = np.array([0.0, 0.0, 0.0])

# 真目标：圆柱体，底面圆心 (0, 60, 0)，半径 5m，高 8m
T_base = np.array([0.0, 60.0, 0.0])
T_r = 5.0
T_h = 8.0

# 导弹当前位置 —— 在空中！Z 明显高于地面
M_pos = np.array([80.0, 0.0, 48.0])

# 烟幕弹起爆位置 —— 悬浮在导弹与真目标之间的空中
smoke_center = np.array([35.0, 30.0, 28.0])
smoke_r = 6.0  # 有效半径

# ============================================================
# 2. 创建图形
# ============================================================
fig = plt.figure(figsize=(16, 10))  # 加宽以容纳右侧图例
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=18, azim=-48)

# ---- 坐标轴范围 ----
x_lim = (0, 100)
y_lim = (0, 80)
z_lim = (0, 70)
ax.set_xlim(*x_lim)
ax.set_ylim(*y_lim)
ax.set_zlim(*z_lim)

# ★ 关键：让三个坐标轴的单位长度相等，球体才不会被压扁 ★
dx = x_lim[1] - x_lim[0]
dy = y_lim[1] - y_lim[0]
dz = z_lim[1] - z_lim[0]
ax.set_box_aspect([dx, dy, dz])  # 数据单位等长 → 球体 = 球体

ax.set_xlabel('X (m)', labelpad=8, fontsize=12)
ax.set_ylabel('Y (m)', labelpad=8, fontsize=12)
ax.set_zlabel('Z (m)', labelpad=8, fontsize=12)

# ============================================================
# 3. 地面参考网格
# ============================================================
xx = np.linspace(0, 100, 20)
yy = np.linspace(0, 80, 17)
XX, YY = np.meshgrid(xx, yy)
ZZ = np.zeros_like(XX)
ax.plot_surface(XX, YY, ZZ, color='#EAECEE', alpha=0.3, edgecolor='#BDC3C7', linewidth=0.15, shade=False)

# ============================================================
# 4. 原点 — 假目标（诱饵）★ 在坐标轴交汇角 ★
# ============================================================
ax.scatter(*O, color='#E67E22', s=280, marker='*', edgecolors='#A04000', linewidth=1.5, zorder=10)
ax.text(O[0] - 12, O[1] - 8, O[2] + 3,
        '假目标(诱饵)\nO(0,0,0)', fontsize=9, ha='center', color='#A04000', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEBD0', alpha=0.9))

# ============================================================
# 5. 真目标 — 红色圆柱体
# ============================================================
def draw_cylinder(ax, base, r, h, color='#E74C3C', alpha=0.8):
    cx, cy, cz = base
    n_theta, n_z, n_r = 50, 3, 12
    theta = np.linspace(0, 2*np.pi, n_theta)

    # 侧面
    t_grid, z_grid = np.meshgrid(theta, np.linspace(cz, cz+h, n_z))
    ax.plot_surface(cx + r*np.cos(t_grid), cy + r*np.sin(t_grid), z_grid,
                    color=color, alpha=alpha, shade=True, edgecolor='#7B241C', linewidth=0.12)

    # 顶面 + 底面（用实心圆盘）
    rr, tt = np.meshgrid(np.linspace(0, r, n_r), np.linspace(0, 2*np.pi, n_theta))
    for zz in [cz, cz + h]:
        ax.plot_surface(cx + rr*np.cos(tt), cy + rr*np.sin(tt), np.full_like(rr, zz),
                        color=color, alpha=alpha, shade=True, edgecolor='#7B241C', linewidth=0.08)

draw_cylinder(ax, T_base, T_r, T_h, color='#E74C3C', alpha=0.78)

ax.text(T_base[0] - 15, T_base[1] + 2, T_base[2] + T_h + 4,
        '真目标\n(圆柱)', fontsize=10, ha='center', color='#C0392B', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#FDEDEC', alpha=0.9))

# ============================================================
# 6. 导弹 — 三角锥体 + 轨迹线
# ============================================================
# 导弹用三角锥表示（弹头朝原点）
ax.scatter(*M_pos, color='black', s=140, marker='^', zorder=9)
ax.text(M_pos[0] + 5, M_pos[1] - 10, M_pos[2] + 5,
        '来袭导弹', fontsize=10, ha='center', color='black', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#F2F3F4', alpha=0.9))

# 导弹飞行方向箭头（指向假目标 = 原点）
v_dir = O - M_pos
v_dir = v_dir / np.linalg.norm(v_dir)
arrow_len = 20
arrow_end = M_pos + v_dir * arrow_len
ax.quiver(M_pos[0], M_pos[1], M_pos[2],
          v_dir[0]*arrow_len, v_dir[1]*arrow_len, v_dir[2]*arrow_len,
          color='black', linewidth=1.8, arrow_length_ratio=0.2, alpha=0.7)

# 导弹速度标注
ax.text(M_pos[0] - 10, M_pos[1] + 5, M_pos[2] + 5,
        'v=300 m/s', fontsize=8, color='gray')

# ============================================================
# 7. 烟幕云团 — 球体 ★ 核心元素 ★
# ============================================================
def draw_sphere(ax, center, r, color='#5DADE2', alpha=0.4, n=50):
    cx, cy, cz = center
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = cx + r * np.outer(np.cos(u), np.sin(v))
    y = cy + r * np.outer(np.sin(u), np.sin(v))
    z = cz + r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor='#2E86C1', linewidth=0.06, shade=True)

draw_sphere(ax, smoke_center, smoke_r, color='#5DADE2', alpha=0.42)

# 球体轮廓线（经纬线，增强立体感）
for fraction in [1.0, 0.6, 0.2]:
    phi_val = np.arccos(1 - 2*fraction) if fraction < 0.99 else np.pi/2
    # 纬度圈
    r_lat = smoke_r * np.sin(phi_val)
    z_lat = smoke_center[2] + smoke_r * np.cos(phi_val)
    theta_c = np.linspace(0, 2*np.pi, 80)
    ax.plot(smoke_center[0] + r_lat*np.cos(theta_c),
            smoke_center[1] + r_lat*np.sin(theta_c),
            np.full_like(theta_c, z_lat),
            '-', color='#2E86C1', linewidth=0.5, alpha=0.5, zorder=5)

# 经线
for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
    vv = np.linspace(0, np.pi, 40)
    ax.plot(smoke_center[0] + smoke_r*np.cos(angle)*np.sin(vv),
            smoke_center[1] + smoke_r*np.sin(angle)*np.sin(vv),
            smoke_center[2] + smoke_r*np.cos(vv),
            '-', color='#2E86C1', linewidth=0.5, alpha=0.5, zorder=5)

# 球心标注
ax.scatter(*smoke_center, color='#1A5276', s=40, marker='.', zorder=8)
ax.text(smoke_center[0] + 2, smoke_center[1] + 12, smoke_center[2] + 2,
        '烟幕云团\n(球体, r=10m)', fontsize=10, ha='center', color='#1A5276', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#D6EAF8', alpha=0.9))

# 下沉方向箭头
ax.quiver(smoke_center[0], smoke_center[1], smoke_center[2] - smoke_r - 3,
          0, 0, -8, color='#2E86C1', linewidth=1.2, arrow_length_ratio=0.25, alpha=0.7)
ax.text(smoke_center[0] + 5, smoke_center[1], smoke_center[2] - smoke_r - 6,
        '3 m/s↓', fontsize=8, color='#2E86C1', style='italic')

# ============================================================
# 8. 视线遮蔽几何 — 切线锥模型 ★ 核心 ★
# ============================================================
# 原理：从导弹看向烟幕球体，所有与球体相交（或相切）的视线被阻断
# 切线构成一个圆锥面——遮蔽锥，锥内区域 = 被遮蔽区

# 8a. 计算切线锥几何参数
M = M_pos.astype(float)
C = smoke_center.astype(float)
r = smoke_r

# 导弹到球心向量
MC = C - M
d = np.linalg.norm(MC)            # 导弹到球心距离
u = MC / d                         # 单位方向 (导弹→球心)

if d > r:
    # 半锥角
    alpha = np.arcsin(r / d)
    # 导弹到切点的距离（沿锥面）
    L_tan = np.sqrt(d**2 - r**2)
    # 切点圆的圆心（在 MC 连线上）
    C_tan_circle = M + u * (d - r * np.sin(alpha))
    # 切点圆半径
    r_tan_circle = r * np.cos(alpha)

    # 建立垂直坐标系（u 的法平面）
    # 找两个与 u 正交的单位向量
    if abs(u[0]) < 0.9:
        e1 = np.array([1, 0, 0])
    else:
        e1 = np.array([0, 1, 0])
    e1 = e1 - np.dot(e1, u) * u
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(u, e1)

    # 8b. 绘制切线（16条，均匀分布）
    n_tan_lines = 16
    angles = np.linspace(0, 2 * np.pi, n_tan_lines + 1)[:-1]

    for theta in angles:
        # 切点位置
        tan_pt = C_tan_circle + r_tan_circle * (np.cos(theta) * e1 + np.sin(theta) * e2)
        # 验证：tan_pt 到 M 的距离应 ≈ L_tan
        # 从导弹到切点画线（虚线）
        ax.plot([M[0], tan_pt[0]], [M[1], tan_pt[1]], [M[2], tan_pt[2]],
                '-', color='#E67E22', linewidth=0.8, alpha=0.55, zorder=3)

    # 8c. 绘制切点圆（球面上的切点轨迹）
    theta_c = np.linspace(0, 2 * np.pi, 100)
    circle_pts = C_tan_circle[:, None] + r_tan_circle * (
        np.cos(theta_c)[None, :] * e1[:, None] +
        np.sin(theta_c)[None, :] * e2[:, None])
    ax.plot(circle_pts[0], circle_pts[1], circle_pts[2],
            '-', color='#E67E22', linewidth=2.0, alpha=0.85, zorder=6,
            label='切点圆')

    # 8d. 绘制半透明锥面（遮蔽锥）
    n_cone = 12
    n_depth = 2
    for i in range(n_cone):
        th1 = angles[i]
        th2 = angles[(i + 1) % n_cone]
        p1 = C_tan_circle + r_tan_circle * (np.cos(th1) * e1 + np.sin(th1) * e2)
        p2 = C_tan_circle + r_tan_circle * (np.cos(th2) * e1 + np.sin(th2) * e2)
        # 三角形面片：M → p1 → p2
        tri = np.array([M, p1, p2])
        poly = Poly3DCollection([tri], alpha=0.10, color='#F39C12',
                                 edgecolor='none', zorder=1)
        ax.add_collection3d(poly)

    # 8e. 视线到真目标（穿过球体的中心视线用于对比）
    T_center = T_base + np.array([0.0, 0.0, T_h / 2])
    v_center = T_center - M
    # 中心视线与球体的交点
    w_center = M - C
    a_c = np.dot(v_center, v_center)
    b_c = 2 * np.dot(w_center, v_center)
    c_c = np.dot(w_center, w_center) - r**2
    disc_c = b_c**2 - 4 * a_c * c_c

    if disc_c >= 0:
        t1_c = max(0, (-b_c - np.sqrt(disc_c)) / (2 * a_c))
        t2_c = min(1, (-b_c + np.sqrt(disc_c)) / (2 * a_c))
        if t1_c < t2_c:
            pt1_c = M + t1_c * v_center
            pt2_c = M + t2_c * v_center
            # 球体之前：红色虚线
            ax.plot([M[0], pt1_c[0]], [M[1], pt1_c[1]], [M[2], pt1_c[2]],
                    '--', color='#E74C3C', linewidth=1.0, alpha=0.4, zorder=3)
            # 球体内：绿色加粗（被遮蔽）
            ax.plot([pt1_c[0], pt2_c[0]], [pt1_c[1], pt2_c[1]], [pt1_c[2], pt2_c[2]],
                    '-', color='#27AE60', linewidth=4.0, alpha=0.85, zorder=7)
            # 球体之后到目标：红色虚线
            ax.plot([pt2_c[0], T_center[0]], [pt2_c[1], T_center[1]], [pt2_c[2], T_center[2]],
                    '--', color='#E74C3C', linewidth=1.0, alpha=0.4, zorder=3)

    # 8f. 判断真目标是否在遮蔽锥内
    # 目标上、中、下三点
    T_pts_check = [T_base + np.array([0, 0, T_h]),
                   T_base + np.array([0, 0, T_h / 2]),
                   T_base + np.array([0, 0, 0])]
    for T_chk in T_pts_check:
        v_chk = T_chk - M
        # 该方向与锥轴 u 的夹角
        cos_angle = np.dot(v_chk, u) / np.linalg.norm(v_chk)
        angle_chk = np.arccos(np.clip(cos_angle, -1, 1))
        if angle_chk < alpha:
            # 在遮蔽锥内 → 被完全遮蔽！
            ax.scatter(*T_chk, color='#27AE60', s=35, marker='o', zorder=10, linewidth=0.8)

    # 8g. 遮蔽锥标注
    mid_cone = M + u * (d * 0.55)
    ax.text(mid_cone[0] - 10, mid_cone[1] + 5, mid_cone[2] + 5,
            '遮蔽锥\n(锥内视线\n均被阻断)', fontsize=9, ha='center', color='#D35400', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEBD0', alpha=0.9,
                      edgecolor='#E67E22'))

else:
    # 导弹在球体内（极端情况）
    ax.text(50, 40, 25, '导弹在\n烟幕内部', fontsize=12, color='red', fontweight='bold')

# ============================================================
# 9. 图例
# ============================================================
legend_elements = [
    mpatches.Patch(color='#E74C3C', alpha=0.7, label='真目标 (圆柱体)'),
    mpatches.Patch(color='#5DADE2', alpha=0.5, label='烟幕云团 (球体)'),
    plt.Line2D([0],[0], color='black', marker='^', markersize=8, linewidth=0,
               label='来袭导弹'),
    plt.Line2D([0],[0], color='#E67E22', linewidth=1.5, label='切线 (遮蔽锥边界)'),
    plt.Line2D([0],[0], color='#27AE60', linewidth=4.0,
               label='中心视线 被遮蔽段'),
    plt.Line2D([0],[0], color='#E74C3C', linewidth=1, linestyle='--',
               label='中心视线 (参考)'),
    mpatches.Patch(color='#F39C12', alpha=0.15, label='遮蔽锥区域'),
    plt.Line2D([0],[0], color='#E67E22', marker='*', markersize=12, linewidth=0,
               label='假目标 (诱饵)'),
]
ax.legend(handles=legend_elements, loc='center left',
          bbox_to_anchor=(1.02, 0.5), fontsize=9,
          framealpha=0.9, edgecolor='gray', ncol=1)

# ============================================================
# 10. 标题与注释
# ============================================================
ax.set_title('图3  烟幕干扰弹遮蔽原理示意图 — 切线锥模型', fontsize=15, fontweight='bold', pad=20)

fig.text(0.5, 0.015,
         '原理：导弹看向烟幕球体的切线构成遮蔽锥；锥内所有视线均被阻断。'
         '当真目标落入遮蔽锥时，导弹无法"看见"真目标。'
         '烟幕云团有效半径10m，起爆后20s内有效，以3m/s匀速下沉。',
         ha='center', fontsize=9.5, color='#555', style='italic')

plt.tight_layout(rect=[0, 0.04, 0.78, 1])  # 右侧留空间给图例
plt.savefig('figure3_schematic.png', dpi=300, facecolor='white', edgecolor='none')
plt.savefig('figure3_schematic.pdf', facecolor='white', edgecolor='none')
print('[OK] figure3_schematic.png / figure3_schematic.pdf 已保存')
plt.show()
