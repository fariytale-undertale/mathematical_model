import numpy as np

def entropy_weight(X):
    """熵权法计算客观权重"""
    # 归一化到 [0,1]
    X_norm = np.zeros_like(X, dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        X_norm[:, j] = (col - col.min()) / (col.max() - col.min() + 1e-10)
    
    # 计算信息熵
    p = X_norm / (X_norm.sum(axis=0) + 1e-10)
    # 避免 log(0)
    p = np.clip(p, 1e-10, 1)
    
    k = 1 / np.log(X.shape[0])
    entropy = -k * np.sum(p * np.log(p), axis=0)
    
    # 计算差异系数和权重
    d = 1 - entropy
    weights = d / d.sum()
    
    return weights


def topsis_complete(decision_matrix, criteria_types, use_entropy_weight=True, weights=None):
    """
    完整版 TOPSIS（支持熵权法自动赋权）
    
    参数:
        decision_matrix: 决策矩阵
        criteria_types: 指标类型
        use_entropy_weight: 是否使用熵权法
        weights: 手动权重（当 use_entropy_weight=False 时使用）
    """
    X = np.array(decision_matrix, dtype=float)
    n = X.shape[1]
    
    # 统一转换为效益型（越大越好），方便熵权计算
    X_transformed = X.copy()
    for j in range(n):
        if criteria_types[j] == -1:  # 成本型取倒数
            X_transformed[:, j] = 1 / (X[:, j] + 1e-10)
    
    # 确定权重
    if use_entropy_weight:
        w = entropy_weight(X_transformed)
        print(f"熵权法计算权重: {w.round(4)}")
    else:
        w = np.array(weights)
        print(f"手动设定权重: {w.round(4)}")
    
    # 向量归一化（用原始矩阵）
    norm = np.sqrt(np.sum(X**2, axis=0))
    R = X / norm
    
    # 加权标准化
    V = R * w
    
    # 确定理想解
    V_plus = np.zeros(n)
    V_minus = np.zeros(n)
    for j in range(n):
        if criteria_types[j] == 1:
            V_plus[j] = np.max(V[:, j])
            V_minus[j] = np.min(V[:, j])
        else:
            V_plus[j] = np.min(V[:, j])
            V_minus[j] = np.max(V[:, j])
    
    # 计算距离
    D_plus = np.sqrt(np.sum((V - V_plus)**2, axis=1))
    D_minus = np.sqrt(np.sum((V - V_minus)**2, axis=1))
    
    # 贴近度
    scores = D_minus / (D_plus + D_minus)
    rankings = np.argsort(scores)[::-1]
    
    return scores, rankings, w, V_plus, V_minus


# ============ 使用示例 ============

print("=" * 60)
print("熵权-TOPSIS 综合评价")
print("=" * 60)

# 数据：5个供应商，4个指标
# [价格(元), 质量评分, 交货期(天), 服务水平]
decision_matrix = [
    [850, 85, 12, 80],
    [700, 70, 10, 75],
    [900, 90, 15, 85],
    [800, 80, 8,  70],
    [750, 75, 11, 78],
]

# 类型: 成本型, 效益型, 成本型, 效益型
criteria_types = [-1, 1, -1, 1]

scores, rankings, weights, v_plus, v_minus = topsis_complete(
    decision_matrix, 
    criteria_types, 
    use_entropy_weight=True
)

print("-" * 60)
print(f"正理想解: {v_plus.round(4)}")
print(f"负理想解: {v_minus.round(4)}")
print("-" * 60)

print("\n排名结果:")
for i, idx in enumerate(rankings):
    print(f"  第{i+1}名: 供应商{idx+1}  (贴近度: {scores[idx]:.4f})")

print("\n各供应商详细得分:")
for i in range(len(scores)):
    print(f"  供应商{i+1}: {scores[i]:.4f}")