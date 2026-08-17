import numpy as np
from analytic_hierarchy_process_withtable import AHPCalculator

# ============================================================
# 旅游方案 AHP 综合评价模型
# 方案层：丽江、厦门、青岛、张家界
# 准则层：交通便利性、旅游资源丰富程度、旅途开销、安全和卫生情况
# ============================================================

alternatives = ['丽江', '厦门', '青岛', '张家界']
criteria = ['交通便利性', '旅游资源丰富程度', '旅途开销', '安全和卫生情况']

# ---------- 准则层判断矩阵（4×4） ----------
# 准则间两两比较（1-9标度法）
# 行/列顺序：交通便利性、旅游资源丰富程度、旅途开销、安全和卫生情况
# 
# 主观设定逻辑：
# - 旅游资源丰富程度 略重要于 交通便利性（3）
# - 旅途开销 明显重要于 交通便利性（5）
# - 安全和卫生情况 介于 略重要 和 明显重要 之间（4）
# - 旅游资源丰富程度 略重要于 旅途开销（3）
# - 安全和卫生情况 略重要于 旅游资源丰富程度（3）
# - 安全和卫生情况 明显重要于 旅途开销（5）
# 
# 注：以下矩阵为示例构造，实际应根据专家打分或问卷调查确定
A_criteria = np.array([
    [1,   1/3, 1/5, 1/4],   # 交通便利性
    [3,   1,   1/3, 1/3],   # 旅游资源丰富程度
    [5,   3,   1,   1/5],   # 旅途开销
    [4,   3,   5,   1]      # 安全和卫生情况
])

# ---------- 方案层判断矩阵（每个准则对应一个 4×4 矩阵） ----------
# 准则1：交通便利性
# 主观设定：厦门 > 青岛 > 丽江 > 张家界
A_traffic = np.array([
    [1,   1/3, 1/2, 2],     # 丽江
    [3,   1,   2,   4],     # 厦门
    [2,   1/2, 1,   3],     # 青岛
    [1/2, 1/4, 1/3, 1]      # 张家界
])

# 准则2：旅游资源丰富程度
# 主观设定：丽江 > 张家界 > 厦门 > 青岛
A_resources = np.array([
    [1,   3,   4,   2],     # 丽江
    [1/3, 1,   2,   1/2],   # 厦门
    [1/4, 1/2, 1,   1/3],   # 青岛
    [1/2, 2,   3,   1]      # 张家界
])

# 准则3：旅途开销（成本型指标，数值越小表示越优，取倒数后比较）
# 主观设定：青岛 < 厦门 < 丽江 < 张家界（青岛最省钱）
A_cost = np.array([
    [1,   1/2, 1/3, 2],     # 丽江
    [2,   1,   1/2, 3],     # 厦门
    [3,   2,   1,   4],     # 青岛
    [1/2, 1/3, 1/4, 1]      # 张家界
])

# 准则4：安全和卫生情况
# 主观设定：厦门 > 青岛 > 丽江 > 张家界
A_safety = np.array([
    [1,   1/3, 1/2, 2],     # 丽江
    [3,   1,   2,   4],     # 厦门
    [2,   1/2, 1,   3],     # 青岛
    [1/2, 1/4, 1/3, 1]      # 张家界
])

# ============================================================
# 计算准则层权重
# ============================================================
ahp_criteria = AHPCalculator(A_criteria, criteria=criteria)
criteria_weights = ahp_criteria.get_weights()
criteria_result = ahp_criteria.result

print("=" * 60)
print("【准则层一致性检验】")
print(f"  λ_max = {criteria_result['lambda_max']:.4f}")
print(f"  CI = {criteria_result['CI']:.4f}")
print(f"  RI = {criteria_result['RI']}")
print(f"  CR = {criteria_result['CR']:.4f}")
print(f"  一致性判断：{'通过 ✓' if criteria_result['consistent'] else '不通过 ✗'}")
print()
print("【准则层权重】")
for c, w in zip(criteria, criteria_weights):
    print(f"  {c}: {w:.4f} ({w*100:.2f}%)")

# ============================================================
# 计算方案层权重（每个准则下各方案的权重）
# ============================================================
matrices = [A_traffic, A_resources, A_cost, A_safety]
alt_weights_matrix = []

print()
print("=" * 60)
print("【方案层一致性检验】")
for i, (mat, crit) in enumerate(zip(matrices, criteria)):
    ahp_alt = AHPCalculator(mat, criteria=alternatives)
    w = ahp_alt.get_weights()
    res = ahp_alt.result
    alt_weights_matrix.append(w)
    print(f" 准则 [{crit}]:")
    print(f"    CR = {res['CR']:.4f} {'通过 ✓' if res['consistent'] else '不通过 ✗'}")
    for alt, wt in zip(alternatives, w):
        print(f"    {alt}: {wt:.4f} ({wt*100:.2f}%)")

alt_weights_matrix = np.array(alt_weights_matrix).T  # 转置为 (4方案 × 4准则)

# ============================================================
# 层次总排序：综合得分 = 方案层权重 × 准则层权重
# ============================================================
final_scores = alt_weights_matrix @ criteria_weights

print()
print("=" * 60)
print("【层次总排序 — 综合得分】")
print()

ranking = sorted(zip(alternatives, final_scores), key=lambda x: x[1], reverse=True)
for rank, (alt, score) in enumerate(ranking, 1):
    bar = "█" * int(score * 50)
    print(f"  第{rank}名: {alt:4s}  综合得分 = {score:.4f}  {bar}")

print()
print("=" * 60)
print(f"【推荐方案】{ranking[0][0]}（综合得分最高）")
print("=" * 60)

# ============================================================
# 可视化：准则层 + 方案层 + 综合得分
# ============================================================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 14))

# ---------- 子图1：准则层权重 ----------
ax1 = axes[0]
ax1.axis('off')
ax1.set_title('准则层权重与一致性检验', fontsize=18, fontweight='bold', pad=15)

criteria_data = []
for c, w in zip(criteria, criteria_weights):
    criteria_data.append([c, f'{w:.4f}', f'{w*100:.2f}%'])
criteria_data.append(['一致性比例 CR', f"{criteria_result['CR']:.4f}", 
                       '通过 ✓' if criteria_result['consistent'] else '不通过 ✗'])

tbl1 = ax1.table(
    cellText=criteria_data, colLabels=['准则', '权重值', '百分比'],
    loc='center', cellLoc='center', bbox=[0.05, 0.1, 0.9, 0.8]
)
tbl1.auto_set_font_size(False)
tbl1.set_fontsize(13)
tbl1.scale(1, 2.2)

for j in range(3):
    tbl1[(0, j)].set_facecolor('#2B61A2')
    tbl1[(0, j)].set_text_props(color='white', fontweight='bold')

for i in range(1, 5):
    if i % 2 == 0:
        for j in range(3):
            tbl1[(i, j)].set_facecolor('#F2F2F2')

cr_color = '#C6EFCE' if criteria_result['consistent'] else '#FFC7CE'
cr_text = '#006100' if criteria_result['consistent'] else '#9C0006'
tbl1[(5, 0)].set_facecolor(cr_color)
tbl1[(5, 1)].set_facecolor(cr_color)
tbl1[(5, 2)].set_facecolor(cr_color)
tbl1[(5, 0)].set_text_props(color=cr_text, fontweight='bold')
tbl1[(5, 1)].set_text_props(color=cr_text, fontweight='bold')
tbl1[(5, 2)].set_text_props(color=cr_text, fontweight='bold')

# ---------- 子图2：方案层各准则得分 ----------
ax2 = axes[1]
ax2.axis('off')
ax2.set_title('方案层各准则得分', fontsize=18, fontweight='bold', pad=15)

alt_data = []
for i, alt in enumerate(alternatives):
    row = [alt]
    for j in range(len(criteria)):
        row.append(f'{alt_weights_matrix[i, j]:.4f}')
    alt_data.append(row)

tbl2 = ax2.table(
    cellText=alt_data, colLabels=['方案'] + criteria,
    loc='center', cellLoc='center', bbox=[0.0, 0.1, 0.95, 0.8]
)
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(12)
tbl2.scale(1, 2.2)

for j in range(5):
    tbl2[(0, j)].set_facecolor('#7281A6')
    tbl2[(0, j)].set_text_props(color='white', fontweight='bold')

for i in range(1, 5):
    if i % 2 == 0:
        for j in range(5):
            tbl2[(i, j)].set_facecolor('#F2F2F2')

# ---------- 子图3：综合得分排名 ----------
ax3 = axes[2]
ax3.axis('off')
ax3.set_title('层次总排序 — 综合得分排名', fontsize=18, fontweight='bold', pad=15)

score_data = []
for rank, (alt, score) in enumerate(ranking, 1):
    score_data.append([f'第{rank}名', alt, f'{score:.4f}', f'{score*100:.2f}%'])

tbl3 = ax3.table(
    cellText=score_data, colLabels=['排名', '方案', '综合得分', '百分比'],
    loc='center', cellLoc='center', bbox=[0.1, 0.15, 0.8, 0.75]
)
tbl3.auto_set_font_size(False)
tbl3.set_fontsize(14)
tbl3.scale(1, 2.5)

for j in range(4):
    tbl3[(0, j)].set_facecolor('#4472C4')
    tbl3[(0, j)].set_text_props(color='white', fontweight='bold')

for i in range(1, 5):
    if i == 1:
        for j in range(4):
            tbl3[(i, j)].set_facecolor('#C6EFCE')
            tbl3[(i, j)].set_text_props(color='#006100', fontweight='bold')
    elif i % 2 == 0:
        for j in range(4):
            tbl3[(i, j)].set_facecolor('#F2F2F2')

plt.tight_layout(pad=3.0)
plt.savefig('tourism_ahp_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()