import numpy as np

def calculate_cr(matrix):
    n = matrix.shape[0]
    # 计算最大特征值
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    lambda_max = np.max(eigenvalues.real)
    
    # 一致性指标 CI
    CI = (lambda_max - n) / (n - 1)
    
    # 随机一致性指标 RI（查表）
    RI_table = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 
                6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = RI_table.get(n, 1.49)
    
    # 一致性比例
    CR = CI / RI
    
    # 权重向量（归一化特征向量）
    idx = np.argmax(eigenvalues.real)
    weights = eigenvectors[:, idx].real
    weights = weights / np.sum(weights)
    
    return {
        'lambda_max': lambda_max,
        'CI': CI,
        'RI': RI,
        'CR': CR,
        'weights': weights,
        'consistent': CR < 0.10
    }

# 示例矩阵
A = np.array([
    [1,   3,   5,   7],
    [1/3, 1,   2,   3],
    [1/5, 1/2, 1, 3/2],
    [1/7, 1/3, 2/3, 1]
])

result = calculate_cr(A)
print(f"最大特征值 λ_max = {result['lambda_max']:.4f}")
print(f"CI = {result['CI']:.4f}")
print(f"RI = {result['RI']}")
print(f"CR = {result['CR']:.4f}")
print(f"权重向量 = {result['weights'].round(4)}")
print(f"一致性通过? {'✅ 是' if result['consistent'] else '❌ 否'}")