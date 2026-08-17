import numpy as np

def genetic_algorithm(func, bounds, pop_size=50, generations=100, 
                      crossover_rate=0.8, mutation_rate=0.1):
    dim = len(bounds)
    # 初始化种群
    pop = np.random.uniform(
        [b[0] for b in bounds], 
        [b[1] for b in bounds], 
        (pop_size, dim)
    )
    
    best_history = []
    
    for gen in range(generations):
        # 评估适应度
        fitness = np.array([func(ind) for ind in pop])
        
        # 记录最优
        best_idx = np.argmax(fitness)
        best_history.append((pop[best_idx], fitness[best_idx]))
        
        # 选择（锦标赛）
        selected = []
        for _ in range(pop_size):
            i, j = np.random.choice(pop_size, 2, replace=False)
            selected.append(pop[i] if fitness[i] > fitness[j] else pop[j])
        selected = np.array(selected)
        
        # 交叉
        offspring = []
        for i in range(0, pop_size, 2):
            p1, p2 = selected[i], selected[i+1] if i+1 < pop_size else selected[0]
            if np.random.rand() < crossover_rate:
                alpha = np.random.rand(dim)
                c1 = alpha * p1 + (1-alpha) * p2
                c2 = alpha * p2 + (1-alpha) * p1
                offspring.extend([c1, c2])
            else:
                offspring.extend([p1, p2])
        offspring = np.array(offspring[:pop_size])
        
        # 变异
        for i in range(pop_size):
            if np.random.rand() < mutation_rate:
                offspring[i] += np.random.normal(0, 0.1, dim)
                offspring[i] = np.clip(offspring[i], 
                    [b[0] for b in bounds], [b[1] for b in bounds])
        
        pop = offspring
    
    return best_history[-1]

# 使用示例
def objective(x):
    return x[0] * np.sin(10 * np.pi * x[0]) + 2

best_x, best_f = genetic_algorithm(objective, [(-1, 2)])
print(f"最优解: x={best_x[0]:.4f}, f(x)={best_f:.4f}")