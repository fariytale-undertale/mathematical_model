"""
TOPSIS 法评价区域科技创新能力
================================
引用 topsis_module 模块完成多属性决策分析

题目：评价 A-J 共10个省份的科技创新能力
指标：经费、研发人员、GDP能耗、高企数量、专利数、科技成果转化率、政策支持强度
"""

#import sys
#sys.path.insert(0, '/mnt/agents/output')  # 模块所在路径

from topsis_withtable import TOPSIS

# ==================== 数据准备 ====================

# 决策矩阵：10个省份 × 7个指标
# 指标顺序：经费、研发人员、GDP能耗、高企数量、专利数、科技成果转化率、政策支持强度
decision_matrix = [
    [2.8, 165, 0.75, 1020, 2300, 55, 85],   # A省
    [3.5, 120, 0.68, 890, 1950, 62, 70],    # B省
    [2.2, 140, 0.91, 1100, 2500, 66, 95],   # C省
    [3.1, 155, 0.74, 980, 2100, 45, 60],    # D省
    [2.5, 180, 0.70, 950, 2400, 60, 80],    # E省
    [3.3, 170, 0.72, 1050, 2350, 58, 75],   # F省
    [2.9, 130, 0.88, 970, 2000, 50, 88],    # G省
    [2.6, 160, 0.76, 930, 2250, 63, 90],    # H省
    [3.0, 150, 0.69, 1000, 2150, 59, 65],   # I省
    [2.4, 145, 0.73, 960, 2050, 61, 78],    # J省
]

# 指标类型：
#   1  = 效益型（越大越好）
#   -1 = 成本型（越小越好）
#   0  = 中间型（越接近目标值越好）
#   2  = 区间型（越落在区间内越好）
criteria_types = [1, 1, -1, 1, 1, 0, 2]

# 中间型指标目标值（仅类型为0的需要提供）
# 科技成果转化率：理想值为60%
target_values = [None, None, None, None, None, 60, None]

# 区间型指标区间（仅类型为2的需要提供）
# 政策支持强度：理想区间为[70, 90]
intervals = [None, None, None, None, None, None, (70, 90)]

# 指标名称和方案名称
criteria_names = ['经费', '研发人员', 'GDP能耗', '高企数量', '专利数', '科技成果转化率', '政策支持强度']
alternative_names = ['A省', 'B省', 'C省', 'D省', 'E省', 'F省', 'G省', 'H省', 'I省', 'J省']


# ==================== 执行 TOPSIS 分析 ====================

# 创建 TOPSIS 模型
model = TOPSIS()

# 拟合数据
model.fit(
    decision_matrix, 
    criteria_types,
    target_values=target_values,
    intervals=intervals,
    criteria_names=criteria_names,
    alternative_names=alternative_names
)

# 打印完整结果汇总
model.summary()

# 获取最优方案
best_name, best_score = model.get_best()
print(f"\n最优方案: {best_name} (贴近度: {best_score:.4f})")

model.plot_bar(save_path='topsis_bar.png')
model.plot_radar(save_path='topsis_radar.png')
model.plot_weights(save_path='topsis_weights.png')
model.plot_full_table(save_path='topsis_result_table.png')

# ==================== 导出 Excel（可选）====================
# model.export_excel('topsis_result.xlsx')