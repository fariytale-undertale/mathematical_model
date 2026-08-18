"""
2024 CUMCM Problem B: Configuration and Data Tables
生产过程中的决策问题 — 参数配置
"""

import numpy as np

# ============================================================
# Problem 1: Sampling Inspection Parameters
# ============================================================
P0_NOMINAL = 0.10          # 标称次品率
ALPHA_CASE1 = 0.05         # 情形(1): 95%信度拒绝 (Type I error)
BETA_CASE2 = 0.10          # 情形(2): 90%信度接收 (Type II error)
P1_ALTERNATIVE = 0.15      # 备择次品率 (可分辨差异 δ = 0.05)
# For Bayesian stopping
PRIOR_ALPHA = 1.0          # Beta先验 α
PRIOR_BETA = 9.0           # Beta先验 β (mean = 0.10)
COST_PER_TEST = 1.0        # 单次检测成本 (相对值)
LOSS_REJECT = 1000.0       # 错误拒收损失
LOSS_ACCEPT = 5000.0       # 错误接收损失

# ============================================================
# Problem 2: Table 1 — 6 scenarios for 2-component assembly
# ============================================================
# Columns: [p1, buy1, test1, p2, buy2, test2, pf, assy, test_f, price, loss_return, disassemble]
# p1, p2: 零配件次品率
# buy1, buy2: 购买单价
# test1, test2: 检测成本
# pf: 成品次品率 (given both components good)
# assy: 装配成本
# test_f: 成品检测成本
# price: 市场售价
# loss_return: 调换损失
# disassemble: 拆解费用

TABLE1_SCENARIOS = [
    # 情形 1
    dict(p1=0.10, buy1=4, test1=2,
         p2=0.10, buy2=18, test2=3,
         pf=0.10, assy=6, test_f=3,
         price=56, loss_return=6, dis=5),
    # 情形 2
    dict(p1=0.20, buy1=4, test1=2,
         p2=0.20, buy2=18, test2=3,
         pf=0.20, assy=6, test_f=3,
         price=56, loss_return=6, dis=5),
    # 情形 3
    dict(p1=0.10, buy1=4, test1=2,
         p2=0.10, buy2=18, test2=3,
         pf=0.10, assy=6, test_f=3,
         price=56, loss_return=30, dis=5),
    # 情形 4
    dict(p1=0.20, buy1=4, test1=1,
         p2=0.20, buy2=18, test2=1,
         pf=0.20, assy=6, test_f=2,
         price=56, loss_return=30, dis=5),
    # 情形 5
    dict(p1=0.10, buy1=4, test1=8,
         p2=0.20, buy2=18, test2=1,
         pf=0.10, assy=6, test_f=2,
         price=56, loss_return=10, dis=5),
    # 情形 6
    dict(p1=0.05, buy1=4, test1=2,
         p2=0.05, buy2=18, test2=3,
         pf=0.05, assy=6, test_f=3,
         price=56, loss_return=10, dis=40),
]

# ============================================================
# Problem 3: Table 2 — 2 processes, 8 components
# ============================================================
# Assembly structure (Figure 1):
#   Components 1,2,3 → Semi-product 1
#   Components 4,5,6 → Semi-product 2
#   Components 7,8   → Semi-product 3
#   Semi-products 1,2,3 → Final product

TABLE2_COMPONENTS = [
    # idx, p, buy, test_cost
    (1, 0.10, 2, 1),
    (2, 0.10, 8, 1),
    (3, 0.10, 12, 2),
    (4, 0.10, 2, 1),
    (5, 0.10, 8, 1),
    (6, 0.10, 12, 2),
    (7, 0.10, 8, 1),
    (8, 0.10, 12, 2),
]

TABLE2_SEMI = [
    # idx, pf, assy_cost, test_cost, dis_cost, input_components
    (1, 0.10, 8, 4, 6, [0, 1, 2]),   # components 1,2,3 → semi 1
    (2, 0.10, 8, 4, 6, [3, 4, 5]),   # components 4,5,6 → semi 2
    (3, 0.10, 8, 4, 6, [6, 7]),      # components 7,8 → semi 3
]

TABLE2_FINAL = dict(
    pf=0.10, assy=8, test=6, dis=10,
    price=200, loss_return=40,
    input_semi=[0, 1, 2],  # semi-products 1,2,3 → final
)

# ============================================================
# Problem 4: Uncertainty parameters
# ============================================================
N_MC_SAMPLES = 2000          # Monte Carlo samples
N_SPRT_SAMPLES_HISTORICAL = 100  # Historical sample size for SPRT estimation
WASSERSTEIN_EPSILON = 0.05   # DRO Wasserstein ball radius
CONFIDENCE_LEVEL = 0.95      # Confidence level for intervals
