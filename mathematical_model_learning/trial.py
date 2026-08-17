import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm

# ========== 中文字体设置 ==========
font_path = r'C:\Windows\Fonsts\msyh.ttc'   # 微软雅黑

try:
    chinese_font = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = chinese_font.get_name()
except:
    # 如果找不到文件，尝试用字体名称
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

  # --- 1. 绘制真目标（圆柱体，半径7m，高10m）---
  # 底面圆心 (0, 200, 0)
z = np.linspace(0, 10, 20)
theta = np.linspace(0, 2*np.pi, 40)
theta_grid, z_grid = np.meshgrid(theta, z)
x_cyl = 0 + 7 * np.cos(theta_grid)
y_cyl = 200 + 7 * np.sin(theta_grid)
ax.plot_surface(x_cyl, y_cyl, z_grid, alpha=0.5, color='green', label='真目标')

  # --- 2. 绘制假目标（原点标记）---
ax.scatter(0, 0, 0, color='red', s=100, marker='*', label='假目标(原点)')

  # --- 3. 绘制导弹轨迹 ---
t = np.linspace(0, 60, 100)
M1_start = np.array([20000, 0, 2000])
M1_dir = np.array([-20000, 0, -2000])
M1_dir = M1_dir / np.linalg.norm(M1_dir) * 300
M1_traj = M1_start + M1_dir * t[:, np.newaxis]
ax.plot(M1_traj[:,0], M1_traj[:,1], M1_traj[:,2], 'r--', linewidth=2, label='M1轨迹')

  # --- 4. 绘制无人机轨迹 ---
FY1_start = np.array([17800, 0, 1800])
FY1_dir = np.array([-1, 0, 0]) * 120  # 朝向原点，120m/s
t_uav = np.linspace(0, 30, 100)
FY1_traj = FY1_start + FY1_dir * t_uav[:, np.newaxis]
ax.plot(FY1_traj[:,0], FY1_traj[:,1], FY1_traj[:,2], 'b-', linewidth=2, label='FY1轨迹')

  # --- 5. 绘制烟幕云团（球体）---
def plot_sphere(ax, center, radius, color='gray', alpha=0.3):
      u = np.linspace(0, 2*np.pi, 30)
      v = np.linspace(0, np.pi, 30)
      x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
      y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
      z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
      ax.plot_surface(x, y, z, color=color, alpha=alpha)

  # 示例：在某个位置画烟幕球体
smoke_center = (15000, 50, 1700)
plot_sphere(ax, smoke_center, 10, color='gray', alpha=0.4)

  # --- 6. 绘制视线（导弹→真目标）---
t_sample = 20  # 某个采样时刻
M_pos = M1_start + M1_dir * t_sample
T_pos = np.array([0, 200, 5])  # 真目标中心
ax.plot([M_pos[0], T_pos[0]], [M_pos[1], T_pos[1]], [M_pos[2], T_pos[2]],
          'y-', linewidth=1.5, alpha=0.8, label='视线')

  # --- 美化 ---
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('烟幕干扰弹投放策略 — 三维场景示意图', fontsize=14)
ax.legend(loc='upper right')

  # 设置等比例（可选）
max_range = 20000
ax.set_xlim(0, 20000)
ax.set_ylim(-5000, 5000)
ax.set_zlim(0, 3000)

  # 调整视角
ax.view_init(elev=20, azim=-60)
plt.tight_layout()
plt.savefig('3d_scene.png', dpi=300)
plt.show()