import numpy as np
import math

def simulated_annealing(func, bounds, T_init=1000, T_min=1e-8, 
                        alpha=0.995, iterations_per_T=100):
    """
    模拟退火算法
    func: 目标函数（求最小值）
    bounds: [(min, max), ...] 各维度边界
    """
    dim = len(bounds)
    
    # 初始化随机解
    x = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
    best_x = x.copy()
    best_energy = func(x)
    
    T = T_init
    history = [(x.copy(), best_energy, T)]
    
    while T > T_min:
        for _ in range(iterations_per_T):
            # 生成邻域解（高斯扰动）
            x_new = x + np.random.normal(0, T/10, dim)
            x_new = np.clip(x_new, [b[0] for b in bounds], [b[1] for b in bounds])
            
            energy_new = func(x_new)
            delta_E = energy_new - func(x)
            
            # Metropolis 准则
            if delta_E < 0 or np.random.random() < math.exp(-delta_E / T):
                x = x_new.copy()
                if energy_new < best_energy:
                    best_energy = energy_new
                    best_x = x_new.copy()
                    
        
        T *= alpha  # 降温
        history.append((x.copy(), best_energy, T))
    
    return best_x, best_energy, history

# ========== 使用示例 ==========

# 1. 多峰函数优化（有多个局部最小值）
def rastrigin(x):
    """Rastrigin函数，全局最小值在(0,0)，值为0"""
    A = 10
    return A * len(x) + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)

# 2. 求解
best_x, best_f, history = simulated_annealing(
    rastrigin, 
    bounds=[(-5.12, 5.12), (-5.12, 5.12)],
    T_init=100,
    alpha=0.99
)

print(f"最优解: x = [{best_x[0]:.6f}, {best_x[1]:.6f}]")
print(f"最优值: f(x) = {best_f:.6f}")