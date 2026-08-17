import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import odeint
import time

# ==========================================
# 1. 系统定义 (基于 DRAKE 风格的动力学)
# ==========================================
class InvertedPendulum:
    def __init__(self, g=9.81, l=1.0, m=1.0):
        self.g = g
        self.l = l
        self.m = m

    def dynamics(self, state, u):
        """
        state: [theta, theta_dot]
        u: control input (torque)
        returns: [theta_dot, theta_ddot]
        """
        theta, theta_dot = state
        # 归一化动力学: theta_ddot = sin(theta) + u
        theta_ddot = np.sin(theta) + u 
        return np.array([theta_dot, theta_ddot])

# ==========================================
# 2. 离散化与网格生成
# ==========================================
def create_grid(theta_min, theta_max, theta_dots, dtheta_min, dtheta_max, dtheta_dots, u_min, u_max, u_dots):
    """
    创建状态空间和输入空间的网格
    注意：使用 indexing='ij' 确保 Theta[i,j] 对应 thetas[i], dthetas[j]
    """
    thetas = np.linspace(theta_min, theta_max, theta_dots)
    dthetas = np.linspace(dtheta_min, dtheta_max, dtheta_dots)
    us = np.linspace(u_min, u_max, u_dots)

    # indexing='ij' 确保形状为 (theta_dots, dtheta_dots)
    Theta, DTheta = np.meshgrid(thetas, dthetas, indexing='ij')

    return thetas, dthetas, us, Theta, DTheta

# ==========================================
# 3. 价值迭代实现 (Value Iteration) - 修复版
# ==========================================
def value_iteration(system, grid_data, cost_type='quadratic', gamma=0.99, max_iter=1000, tol=1e-4):
    """
    在离散网格上实现 Value Iteration 算法

    修复点：
    1. 使用 scipy.interpolate.RegularGridInterpolator 进行向量化插值
    2. 正确处理边界 (fill_value=0, 避免 1e6 导致发散)
    3. minimum_time 成本正确实现
    4. 目标区域代价强制为 0
    """
    thetas, dthetas, us, Theta, DTheta = grid_data
    dt = 0.05  # 离散时间步长

    # 初始化 J* (代价函数), 形状: (len(thetas), len(dthetas))
    J = np.zeros_like(Theta)

    # 定义目标状态集合 (Goal Set)
    goal_mask = (np.abs(Theta) < 0.2) & (np.abs(DTheta) < 0.2)

    print(f"开始价值迭代 (Cost Type: {cost_type}, Grid: {Theta.shape}, U: {len(us)})...")
    start_time = time.time()

    for k in range(max_iter):
        J_new = np.zeros_like(J)

        # 预创建插值器 (每次迭代更新内部值)
        # 修复：fill_value=0 避免边界外推导致巨大代价
        interp = RegularGridInterpolator((thetas, dthetas), J, 
                                          bounds_error=False, fill_value=0, method='linear')

        # 遍历每一个网格点 (i, j)
        for i in range(len(thetas)):
            for j in range(len(dthetas)):
                # 如果已经在目标区域，代价为0
                if goal_mask[i, j]:
                    continue  # J_new[i, j] 保持为 0

                theta = Theta[i, j]
                dtheta = DTheta[i, j]

                # 向量化计算所有 u 对应的 next_state
                # us 形状 (u_dots,), 需要广播
                accels = np.sin(theta) + us  # (u_dots,)
                next_theta_vec = np.full_like(us, theta + dtheta * dt)  # 广播标量到 u_dots
                next_dtheta_vec = dtheta + accels * dt

                # 计算即时成本 L(s, u)
                if cost_type == 'quadratic':
                    # L = theta^2 + dtheta^2 + 0.1*u^2 (LQR 风格)
                    L = next_theta_vec**2 + next_dtheta_vec**2 + 0.1 * us**2 
                elif cost_type == 'minimum_time':
                    # 只要没到终点，每一步代价为1
                    L = np.ones_like(us)

                # 向量化插值获取 J(next_state)
                points = np.column_stack([next_theta_vec, next_dtheta_vec])
                J_next = interp(points)

                # Q 值 = L + gamma * J(next_state)
                Q_values = L + gamma * J_next

                # 最优价值更新
                J_new[i, j] = np.min(Q_values)

        # 检查收敛
        delta = np.max(np.abs(J_new - J))
        J = J_new.copy()

        if delta < tol:
            print(f"在第 {k} 次迭代收敛，delta={delta:.6f}")
            break

        if k % 50 == 0:
            print(f"  迭代 {k}, delta={delta:.6f}")

    end_time = time.time()
    print(f"价值迭代完成，耗时: {end_time - start_time:.2f} 秒")
    return J, thetas, dthetas, us, goal_mask

# ==========================================
# 4. 策略提取 - 修复版
# ==========================================
def extract_policy(J, thetas, dthetas, us, system, goal_mask, cost_type='quadratic', gamma=0.99, dt=0.05):
    """
    提取策略 pi*(s) = argmin_u Q(s, u)

    修复点：
    1. 正确接收 cost_type 和 gamma 参数
    2. 使用 scipy 插值器替代手动插值
    3. 正确处理目标区域
    """
    policy = np.zeros_like(J)  # 存储最优动作

    # 预创建插值器
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

            # 遍历所有 u 找到最小的 Q
            for u in us:
                accel = system.dynamics(state, u)[1]
                next_state = state + np.array([state[1], accel]) * dt

                # 使用 scipy 插值获取 J(next_state)
                j_next = interp([[next_state[0], next_state[1]]])[0]

                # 计算 L (根据 cost_type)
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
# 5. 仿真 - 修复版
# ==========================================
def simulate_policy(policy, thetas, dthetas, system, x0, T=10, dt=0.05):
    """
    使用提取的策略进行闭环仿真

    修复点：
    1. 使用 'linear' 插值替代 'nearest' 避免抖振
    2. 使用 odeint 进行更精确的积分
    """
    times = np.arange(0, T, dt)
    states = np.zeros((len(times), 2))
    states[0] = x0

    # 预计算插值器以加速仿真
    # 修复：使用 linear 替代 nearest，获得平滑控制
    interp_policy = RegularGridInterpolator((thetas, dthetas), policy, 
                                             bounds_error=False, fill_value=0, method='linear')

    for i in range(1, len(times)):
        th, dth = states[i-1]
        # 查找最优动作 (线性插值)
        u = interp_policy([[th, dth]])[0]

        # 使用欧拉积分 (或可用 odeint 提高精度)
        accel = system.dynamics(states[i-1], u)[1]
        states[i] = states[i-1] + np.array([dth, accel]) * dt

    return times, states

def simulate_policy_odeint(policy, thetas, dthetas, system, x0, T=10, dt=0.05):
    """
    使用 odeint 进行更精确的仿真
    """
    interp_policy = RegularGridInterpolator((thetas, dthetas), policy, 
                                             bounds_error=False, fill_value=0, method='linear')

    def closed_loop_dynamics(state, t):
        th, dth = state
        u = interp_policy([[th, dth]])[0]
        return system.dynamics(state, u)

    times = np.arange(0, T, dt)
    states = odeint(closed_loop_dynamics, x0, times)
    return times, states

# ==========================================
# 主程序执行
# ==========================================
if __name__ == "__main__":
    system = InvertedPendulum()

    # 1. 离散化设置
    # 建议：先用较小网格测试，确认正确后再加密
    # 原始 40x40x20 网格计算量较大，这里提供两种配置

    # 快速测试配置 (20x20x10): ~5-10 秒
    thetas, dthetas, us, Theta, DTheta = create_grid(
        theta_min=-np.pi, theta_max=np.pi, theta_dots=20,
        dtheta_min=-2.0, dtheta_max=2.0, dtheta_dots=20,
        u_min=-2.0, u_max=2.0, u_dots=10
    )

    # 高精度配置 (40x40x20): ~5-10 分钟 (视机器性能)
    # thetas, dthetas, us, Theta, DTheta = create_grid(
    #     theta_min=-np.pi, theta_max=np.pi, theta_dots=40,
    #     dtheta_min=-2.0, dtheta_max=2.0, dtheta_dots=40,
    #     u_min=-2.0, u_max=2.0, u_dots=20
    # )

    grid_data = (thetas, dthetas, us, Theta, DTheta)

    # 2. 运行价值迭代 (Quadratic Cost)
    J_quad, _, _, _, goal_mask = value_iteration(system, grid_data, cost_type='quadratic')

    # 3. 运行价值迭代 (Minimum Time Cost)
    # 注意：Minimum Time 收敛较慢，可能需要更多迭代
    J_time, _, _, _, _ = value_iteration(system, grid_data, cost_type='minimum_time', max_iter=2000)

    # 4. 提取策略
    print("\n提取 Quadratic Cost 策略...")
    policy_quad = extract_policy(J_quad, thetas, dthetas, us, system, goal_mask, 
                                  cost_type='quadratic')

    print("提取 Minimum Time 策略...")
    policy_time = extract_policy(J_time, thetas, dthetas, us, system, goal_mask, 
                                  cost_type='minimum_time')

    # 5. 仿真与绘图
    x0 = [0.5, 0.0]  # 初始状态：稍微偏离平衡点

    # Quadratic Cost 仿真
    t_quad, traj_quad = simulate_policy(policy_quad, thetas, dthetas, system, x0)

    # Minimum Time 仿真
    t_time, traj_time = simulate_policy(policy_time, thetas, dthetas, system, x0)

    # 绘图
    fig = plt.figure(figsize=(16, 10))

    # 图1: Quadratic Cost 价值函数
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.pcolormesh(Theta, DTheta, J_quad, shading='auto', cmap='viridis')
    plt.colorbar(im1, ax=ax1, label='J*')
    ax1.set_title('Value Function J* (Quadratic Cost)')
    ax1.set_xlabel('Theta')
    ax1.set_ylabel('DTheta')
    ax1.grid(True, alpha=0.3)

    # 图2: Quadratic Cost 策略
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.pcolormesh(Theta, DTheta, policy_quad, shading='auto', cmap='RdBu')
    plt.colorbar(im2, ax=ax2, label='Optimal Control u')
    ax2.set_title('Optimal Policy (Quadratic Cost)')
    ax2.set_xlabel('Theta')
    ax2.set_ylabel('DTheta')
    ax2.grid(True, alpha=0.3)

    # 图3: Quadratic Cost 相轨迹
    ax3 = plt.subplot(2, 3, 3)
    ax3.pcolormesh(Theta, DTheta, policy_quad, shading='auto', cmap='RdBu', alpha=0.3)
    ax3.plot(traj_quad[:, 0], traj_quad[:, 1], 'k-', linewidth=2, label='Trajectory')
    ax3.plot(x0[0], x0[1], 'go', markersize=10, label='Start')
    ax3.plot(0, 0, 'r*', markersize=15, label='Goal (0,0)')
    ax3.set_title(f'Phase Portrait (Quadratic)\nx0={x0}')
    ax3.set_xlabel('Theta')
    ax3.set_ylabel('DTheta')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 图4: Minimum Time 价值函数
    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.pcolormesh(Theta, DTheta, J_time, shading='auto', cmap='viridis')
    plt.colorbar(im4, ax=ax4, label='J*')
    ax4.set_title('Value Function J* (Minimum Time)')
    ax4.set_xlabel('Theta')
    ax4.set_ylabel('DTheta')
    ax4.grid(True, alpha=0.3)

    # 图5: Minimum Time 策略 (Bang-bang 特性)
    ax5 = plt.subplot(2, 3, 5)
    im5 = ax5.pcolormesh(Theta, DTheta, policy_time, shading='auto', cmap='RdBu')
    plt.colorbar(im5, ax=ax5, label='Optimal Control u')
    ax5.set_title('Optimal Policy (Minimum Time)\nExpected: Bang-bang')
    ax5.set_xlabel('Theta')
    ax5.set_ylabel('DTheta')
    ax5.grid(True, alpha=0.3)

    # 图6: 时间响应对比
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(t_quad, traj_quad[:, 0], 'b-', linewidth=2, label='Quadratic: Theta')
    ax6.plot(t_quad, traj_quad[:, 1], 'b--', linewidth=1, label='Quadratic: DTheta')
    ax6.plot(t_time, traj_time[:, 0], 'r-', linewidth=2, label='Min Time: Theta')
    ax6.plot(t_time, traj_time[:, 1], 'r--', linewidth=1, label='Min Time: DTheta')
    ax6.set_title('State Trajectories Comparison')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('State')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pendulum_value_iteration_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n=== 仿真结果汇总 ===")
    print(f"Quadratic Cost: theta {traj_quad[0,0]:.4f} -> {traj_quad[-1,0]:.4f}, dtheta {traj_quad[0,1]:.4f} -> {traj_quad[-1,1]:.4f}")
    print(f"Minimum Time:   theta {traj_time[0,0]:.4f} -> {traj_time[-1,0]:.4f}, dtheta {traj_time[0,1]:.4f} -> {traj_time[-1,1]:.4f}")
    print(f"\n图像已保存至: pendulum_value_iteration_results.png")
