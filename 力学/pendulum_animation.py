import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import odeint
import time

# ==========================================
# 1. 系统定义
# ==========================================
class InvertedPendulum:
    def __init__(self, g=9.81, l=1.0, m=1.0):
        self.g = g
        self.l = l
        self.m = m

    def dynamics(self, state, u):
        theta, theta_dot = state
        theta_ddot = np.sin(theta) + u 
        return np.array([theta_dot, theta_ddot])

# ==========================================
# 2. 离散化与网格生成
# ==========================================
def create_grid(theta_min, theta_max, theta_dots, dtheta_min, dtheta_max, dtheta_dots, u_min, u_max, u_dots):
    thetas = np.linspace(theta_min, theta_max, theta_dots)
    dthetas = np.linspace(dtheta_min, dtheta_max, dtheta_dots)
    us = np.linspace(u_min, u_max, u_dots)
    Theta, DTheta = np.meshgrid(thetas, dthetas, indexing='ij')
    return thetas, dthetas, us, Theta, DTheta

# ==========================================
# 3. 价值迭代
# ==========================================
def value_iteration(system, grid_data, cost_type='quadratic', gamma=0.99, max_iter=1000, tol=1e-4):
    thetas, dthetas, us, Theta, DTheta = grid_data
    dt = 0.05
    J = np.zeros_like(Theta)
    goal_mask = (np.abs(Theta) < 0.2) & (np.abs(DTheta) < 0.2)

    print(f"开始价值迭代 (Cost Type: {cost_type}, Grid: {Theta.shape}, U: {len(us)})...")
    start_time = time.time()

    for k in range(max_iter):
        J_new = np.zeros_like(J)
        interp = RegularGridInterpolator((thetas, dthetas), J, 
                                          bounds_error=False, fill_value=0, method='linear')

        for i in range(len(thetas)):
            for j in range(len(dthetas)):
                if goal_mask[i, j]:
                    continue

                theta = Theta[i, j]
                dtheta = DTheta[i, j]
                accels = np.sin(theta) + us
                next_theta_vec = np.full_like(us, theta + dtheta * dt)
                next_dtheta_vec = dtheta + accels * dt

                if cost_type == 'quadratic':
                    L = next_theta_vec**2 + next_dtheta_vec**2 + 0.1 * us**2
                elif cost_type == 'minimum_time':
                    L = np.ones_like(us)

                points = np.column_stack([next_theta_vec, next_dtheta_vec])
                J_next = interp(points)
                Q_values = L + gamma * J_next
                J_new[i, j] = np.min(Q_values)

        delta = np.max(np.abs(J_new - J))
        J = J_new.copy()

        if delta < tol:
            print(f"在第 {k} 次迭代收敛，delta={delta:.6f}")
            break
        if k % 50 == 0:
            print(f"  迭代 {k}, delta={delta:.6f}")

    print(f"价值迭代完成，耗时: {time.time() - start_time:.2f} 秒")
    return J, thetas, dthetas, us, goal_mask

# ==========================================
# 4. 策略提取
# ==========================================
def extract_policy(J, thetas, dthetas, us, system, goal_mask, cost_type='quadratic', gamma=0.99, dt=0.05):
    policy = np.zeros_like(J)
    interp = RegularGridInterpolator((thetas, dthetas), J, 
                                      bounds_error=False, fill_value=0, method='linear')

    for i in range(len(thetas)):
        for j in range(len(dthetas)):
            if goal_mask[i, j]:
                policy[i, j] = 0
                continue

            state = np.array([thetas[i], dthetas[j]])
            best_u = 0
            min_q = float('inf')

            for u in us:
                accel = system.dynamics(state, u)[1]
                next_state = state + np.array([state[1], accel]) * dt
                j_next = interp([[next_state[0], next_state[1]]])[0]

                if cost_type == 'quadratic':
                    L = next_state[0]**2 + next_state[1]**2 + 0.1*u**2
                elif cost_type == 'minimum_time':
                    L = 1.0

                Q = L + gamma * j_next
                if Q < min_q:
                    min_q = Q
                    best_u = u

            policy[i, j] = best_u

    return policy

# ==========================================
# 5. 仿真
# ==========================================
def simulate_policy(policy, thetas, dthetas, system, x0, T=10, dt=0.05):
    times = np.arange(0, T, dt)
    states = np.zeros((len(times), 2))
    states[0] = x0
    controls = np.zeros(len(times))

    interp_policy = RegularGridInterpolator((thetas, dthetas), policy, 
                                             bounds_error=False, fill_value=0, method='linear')

    for i in range(1, len(times)):
        th, dth = states[i-1]
        u = interp_policy([[th, dth]])[0]
        controls[i-1] = u
        accel = system.dynamics(states[i-1], u)[1]
        states[i] = states[i-1] + np.array([dth, accel]) * dt

    controls[-1] = controls[-2]  # 最后一个控制量
    return times, states, controls

# ==========================================
# 6. 动画演示 - 核心部分
# ==========================================
def create_pendulum_animation(times, states, controls, l=1.0, save_gif=None, save_mp4=None, fps=30):
    """
    创建倒立摆动画

    参数:
        times: 时间数组
        states: 状态轨迹 [theta, theta_dot]
        controls: 控制输入轨迹
        l: 摆杆长度
        save_gif: 保存为GIF的文件名 (如 'animation.gif')
        save_mp4: 保存为MP4的文件名 (如 'animation.mp4')
        fps: 帧率
    """
    theta = states[:, 0]
    theta_dot = states[:, 1]

    # 计算摆杆端点坐标 (theta=0 为竖直向上)
    # x = l * sin(theta), y = l * cos(theta) (以支点为原点，向上为正y)
    x = l * np.sin(theta)
    y = l * np.cos(theta)

    # 创建图形
    fig = plt.figure(figsize=(14, 6))

    # 左侧：倒立摆动画
    ax_pendulum = plt.subplot(1, 2, 1)
    ax_pendulum.set_xlim(-1.5*l, 1.5*l)
    ax_pendulum.set_ylim(-1.5*l, 1.5*l)
    ax_pendulum.set_aspect('equal')
    ax_pendulum.grid(True, alpha=0.3)
    ax_pendulum.set_title('Inverted Pendulum Animation')
    ax_pendulum.set_xlabel('x')
    ax_pendulum.set_ylabel('y')

    # 绘制地面/参考线
    ax_pendulum.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax_pendulum.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # 绘制目标位置 (竖直向上)
    ax_pendulum.plot(0, l, 'g*', markersize=15, label='Goal')

    # 初始化绘图元素
    line, = ax_pendulum.plot([], [], 'o-', linewidth=3, markersize=10, color='blue', label='Pendulum')
    mass, = ax_pendulum.plot([], [], 'ro', markersize=15)
    trail, = ax_pendulum.plot([], [], 'b-', alpha=0.2, linewidth=1)

    # 显示当前状态信息
    info_text = ax_pendulum.text(0.02, 0.98, '', transform=ax_pendulum.transAxes, 
                                  fontsize=10, verticalalignment='top',
                                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax_pendulum.legend(loc='lower left')

    # 右侧：状态和控制时域响应
    ax_plot = plt.subplot(1, 2, 2)
    ax_plot.set_xlim(0, times[-1])
    ax_plot.set_ylim(-3, 3)
    ax_plot.set_xlabel('Time (s)')
    ax_plot.set_ylabel('Value')
    ax_plot.set_title('State & Control Trajectories')
    ax_plot.grid(True, alpha=0.3)

    # 绘制完整轨迹 (淡色)
    ax_plot.plot(times, theta, 'b-', alpha=0.2, label='Theta (target=0)')
    ax_plot.plot(times, theta_dot, 'r-', alpha=0.2, label='Theta_dot')
    ax_plot.plot(times, controls, 'g-', alpha=0.2, label='Control u')

    # 当前时间指示线
    vline = ax_plot.axvline(x=0, color='black', linestyle='--', linewidth=1)

    # 当前值标记点
    point_theta, = ax_plot.plot([], [], 'bo', markersize=8)
    point_dtheta, = ax_plot.plot([], [], 'ro', markersize=8)
    point_control, = ax_plot.plot([], [], 'go', markersize=8)

    ax_plot.legend(loc='upper right')

    # 帧采样 (根据fps和仿真dt计算步长)
    dt = times[1] - times[0]
    frame_step = max(1, int(1 / (fps * dt)))
    frame_indices = range(0, len(times), frame_step)

    # 轨迹历史 (用于尾迹效果)
    trail_length = 20

    def init():
        line.set_data([], [])
        mass.set_data([], [])
        trail.set_data([], [])
        info_text.set_text('')
        vline.set_xdata([0, 0])
        point_theta.set_data([], [])
        point_dtheta.set_data([], [])
        point_control.set_data([], [])
        return line, mass, trail, info_text, vline, point_theta, point_dtheta, point_control

    def update(frame):
        i = frame_indices[frame] if frame < len(frame_indices) else len(times) - 1

        # 更新摆杆
        line.set_data([0, x[i]], [0, y[i]])
        mass.set_data([x[i]], [y[i]])

        # 更新尾迹
        start_idx = max(0, i - trail_length)
        trail.set_data(x[start_idx:i+1], y[start_idx:i+1])

        # 更新状态信息
        info_text.set_text(
            f'Time: {times[i]:.2f}s\n'
            f'Theta: {theta[i]:.3f} rad\n'
            f'Theta_dot: {theta_dot[i]:.3f} rad/s\n'
            f'Control: {controls[i]:.3f} Nm'
        )

        # 更新右侧时域图
        vline.set_xdata([times[i], times[i]])
        point_theta.set_data([times[i]], [theta[i]])
        point_dtheta.set_data([times[i]], [theta_dot[i]])
        point_control.set_data([times[i]], [controls[i]])

        return line, mass, trail, info_text, vline, point_theta, point_dtheta, point_control

    num_frames = len(frame_indices)
    anim = FuncAnimation(fig, update, init_func=init, frames=num_frames,
                         interval=1000/fps, blit=True, repeat=True)

    # 保存动画
    if save_gif:
        print(f"正在保存 GIF: {save_gif} ...")
        writer = PillowWriter(fps=fps)
        anim.save(save_gif, writer=writer)
        print(f"GIF 保存完成: {save_gif}")

    if save_mp4:
        print(f"正在保存 MP4: {save_mp4} ...")
        try:
            writer = FFMpegWriter(fps=fps, metadata=dict(artist='InvertedPendulum'),
                                  bitrate=1800)
            anim.save(save_mp4, writer=writer)
            print(f"MP4 保存完成: {save_mp4}")
        except RuntimeError as e:
            print(f"MP4 保存失败 (可能未安装 ffmpeg): {e}")
            print("提示: 安装 ffmpeg 后重试，或使用 GIF 格式")

    plt.tight_layout()
    plt.show()

    return anim

# ==========================================
# 7. 多初始条件对比动画
# ==========================================
def create_comparison_animation(system, policy, thetas, dthetas, initial_conditions, 
                                 T=10, dt=0.05, l=1.0, save_gif=None):
    """
    对比多个初始条件下的倒立摆响应
    """
    # 预计算所有轨迹
    all_states = []
    all_controls = []
    colors = ['blue', 'red', 'green', 'purple', 'orange']

    for x0 in initial_conditions:
        _, states, controls = simulate_policy(policy, thetas, dthetas, system, x0, T, dt)
        all_states.append(states)
        all_controls.append(controls)

    times = np.arange(0, T, dt)

    fig = plt.figure(figsize=(14, 6))

    # 左侧：多摆对比动画
    ax_pendulum = plt.subplot(1, 2, 1)
    ax_pendulum.set_xlim(-1.5*l, 1.5*l)
    ax_pendulum.set_ylim(-1.5*l, 1.5*l)
    ax_pendulum.set_aspect('equal')
    ax_pendulum.grid(True, alpha=0.3)
    ax_pendulum.set_title('Multiple Initial Conditions')
    ax_pendulum.set_xlabel('x')
    ax_pendulum.set_ylabel('y')
    ax_pendulum.plot(0, l, 'g*', markersize=15, label='Goal')

    lines = []
    masses = []
    for idx, x0 in enumerate(initial_conditions):
        color = colors[idx % len(colors)]
        line, = ax_pendulum.plot([], [], 'o-', linewidth=2, markersize=6, 
                                  color=color, label=f'x0={x0}')
        mass, = ax_pendulum.plot([], [], 'o', markersize=10, color=color)
        lines.append(line)
        masses.append(mass)

    ax_pendulum.legend(loc='lower left', fontsize=8)

    # 右侧：Theta 对比
    ax_plot = plt.subplot(1, 2, 2)
    ax_plot.set_xlim(0, T)
    ax_plot.set_xlabel('Time (s)')
    ax_plot.set_ylabel('Theta (rad)')
    ax_plot.set_title('Theta Convergence Comparison')
    ax_plot.grid(True, alpha=0.3)
    ax_plot.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Target')

    # 绘制完整轨迹
    for idx, (states, x0) in enumerate(zip(all_states, initial_conditions)):
        color = colors[idx % len(colors)]
        ax_plot.plot(times, states[:, 0], alpha=0.3, color=color)

    vline = ax_plot.axvline(x=0, color='black', linestyle='--', linewidth=1)
    points = [ax_plot.plot([], [], 'o', markersize=8, color=colors[i % len(colors)])[0] 
              for i in range(len(initial_conditions))]
    ax_plot.legend()

    fps = 30
    frame_step = max(1, int(1 / (fps * dt)))
    frame_indices = range(0, len(times), frame_step)

    def init():
        for line, mass in zip(lines, masses):
            line.set_data([], [])
            mass.set_data([], [])
        vline.set_xdata([0, 0])
        for p in points:
            p.set_data([], [])
        return lines + masses + [vline] + points

    def update(frame):
        i = frame_indices[frame] if frame < len(frame_indices) else len(times) - 1

        for idx, states in enumerate(all_states):
            theta = states[:, 0]
            x = l * np.sin(theta)
            y = l * np.cos(theta)
            lines[idx].set_data([0, x[i]], [0, y[i]])
            masses[idx].set_data([x[i]], [y[i]])
            points[idx].set_data([times[i]], [theta[i]])

        vline.set_xdata([times[i], times[i]])
        return lines + masses + [vline] + points

    anim = FuncAnimation(fig, update, init_func=init, frames=len(frame_indices),
                         interval=1000/fps, blit=True, repeat=True)

    if save_gif:
        print(f"正在保存对比 GIF: {save_gif} ...")
        writer = PillowWriter(fps=fps)
        anim.save(save_gif, writer=writer)
        print(f"对比 GIF 保存完成: {save_gif}")

    plt.tight_layout()
    plt.show()
    return anim

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    system = InvertedPendulum()

    # 离散化 (使用较小网格以便快速计算)
    thetas, dthetas, us, Theta, DTheta = create_grid(
        theta_min=-np.pi, theta_max=np.pi, theta_dots=20,
        dtheta_min=-2.0, dtheta_max=2.0, dtheta_dots=20,
        u_min=-2.0, u_max=2.0, u_dots=10
    )
    grid_data = (thetas, dthetas, us, Theta, DTheta)

    # 运行价值迭代
    J_quad, _, _, _, goal_mask = value_iteration(system, grid_data, cost_type='quadratic')

    # 提取策略
    print("\n提取策略...")
    policy_quad = extract_policy(J_quad, thetas, dthetas, us, system, goal_mask, 
                                  cost_type='quadratic')

    # 仿真
    x0 = [0.5, 0.0]
    T, dt = 10, 0.05
    times, states, controls = simulate_policy(policy_quad, thetas, dthetas, system, x0, T, dt)

    print(f"\n仿真完成: theta {states[0,0]:.4f} -> {states[-1,0]:.4f}")

    # ============ 动画演示 ============
    print("\n=== 生成动画 ===")

    # 1. 单轨迹动画 (实时显示 + 保存GIF)
    anim = create_pendulum_animation(
        times, states, controls, 
        l=1.0,
        save_gif='pendulum_animation.gif',  # 保存为GIF
        # save_mp4='pendulum_animation.mp4',  # 需要ffmpeg
        fps=30
    )

    # 2. 多初始条件对比动画
    # initial_conditions = [[0.5, 0], [1.0, 0], [1.5, 0], [2.0, 0], [-1.0, 0.5]]
    # anim2 = create_comparison_animation(
    #     system, policy_quad, thetas, dthetas, 
    #     initial_conditions, T=10, dt=0.05,
    #     save_gif='pendulum_comparison.gif'
    # )

    print("\n动画演示完成！")
    print("- 实时动画窗口已弹出")
    print("- GIF 已保存至: pendulum_animation.gif")
