import numpy as np
from typing import Callable, Tuple, List

class PSO:
    def __init__(self, func, dim, bounds, n_particles=30, max_iter=100,
                 w='linear', c1=2.0, c2=2.0, v_max_ratio=0.2):
        self.func = func
        self.dim = dim
        self.bounds = np.array(bounds)
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w_strategy = w
        self.c1, self.c2 = c1, c2
        
        self.x_range = self.bounds[:, 1] - self.bounds[:, 0]
        self.v_max = v_max_ratio * self.x_range
        
        self._init_particles()
        self.history = []
    
    def _init_particles(self):
        # 随机初始化位置
        self.positions = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], 
            (self.n_particles, self.dim)
        )
        # 随机初始化速度
        self.velocities = np.random.uniform(
            -self.v_max, self.v_max, 
            (self.n_particles, self.dim)
        )
        # 个体最优
        self.pbest_pos = self.positions.copy()
        self.pbest_val = np.array([self.func(p) for p in self.positions])
        # 全局最优
        self.gbest_idx = np.argmin(self.pbest_val)
        self.gbest_pos = self.pbest_pos[self.gbest_idx].copy()
        self.gbest_val = self.pbest_val[self.gbest_idx]
    
    def optimize(self, verbose=True):
        for i in range(self.max_iter):
            # 1. 速度更新
            w = 0.9 - 0.5 * i / self.max_iter if self.w_strategy == 'linear' else self.w_strategy
            r1 = np.random.random((self.n_particles, self.dim))
            r2 = np.random.random((self.n_particles, self.dim))
            
            self.velocities = (w * self.velocities 
                             + self.c1 * r1 * (self.pbest_pos - self.positions)
                             + self.c2 * r2 * (self.gbest_pos - self.positions))
            self.velocities = np.clip(self.velocities, -self.v_max, self.v_max)
            
            # 2. 位置更新
            self.positions += self.velocities
            
            # 3. 边界处理（反射法）
            for d in range(self.dim):
                lo, hi = self.bounds[d]
                mask_low = self.positions[:, d] < lo
                self.positions[mask_low, d] = 2*lo - self.positions[mask_low, d]
                self.velocities[mask_low, d] *= -0.5
                mask_high = self.positions[:, d] > hi
                self.positions[mask_high, d] = 2*hi - self.positions[mask_high, d]
                self.velocities[mask_high, d] *= -0.5
            
            # 4. 更新最优
            current_vals = np.array([self.func(p) for p in self.positions])
            improved = current_vals < self.pbest_val
            self.pbest_pos[improved] = self.positions[improved].copy()
            self.pbest_val[improved] = current_vals[improved]
            
            best_idx = np.argmin(self.pbest_val)
            if self.pbest_val[best_idx] < self.gbest_val:
                self.gbest_pos = self.pbest_pos[best_idx].copy()
                self.gbest_val = self.pbest_val[best_idx]
            
            self.history.append(self.gbest_val)
        
        return self.gbest_pos, self.gbest_val, self.history