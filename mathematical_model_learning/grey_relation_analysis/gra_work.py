import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 引用灰色关联分析类（头文件）
from grey_relation_analysis import GreyRelationalAnalysis
import matplotlib.font_manager as fm

# ========== 中文字体设置 ==========
# Windows 系统常用路径，根据你的系统选择：
font_path = r'C:\Windows\Fonts\simhei.ttf'  # 黑体
# font_path = r'C:\Windows\Fonts\msyh.ttc'   # 微软雅黑（推荐）

try:
    chinese_font = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = chinese_font.get_name()
except:
    # 如果找不到文件，尝试用字体名称
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    
plt.rcParams['axes.unicode_minus'] = False
# ==================== 任务：公司评估灰色关联分析 ====================

# 1. 构建数据
data = {
    '公司名称': ['A公司', 'B公司', 'C公司', 'D公司', 'E公司'],
    '公司业绩': [80, 70, 85, 90, 75],
    '市场占有率': [70, 65, 80, 75, 60],
    '技术创新能力': [85, 78, 90, 80, 70],
    '客户满意度': [75, 80, 70, 85, 80]
}
df = pd.DataFrame(data)
print("原始数据")
print(df.to_string(index=False))
print()

# 2. 确定理想公司（各指标取最大值）
metrics = ['公司业绩', '市场占有率', '技术创新能力', '客户满意度']
ideal = df[metrics].max().values
print("理想公司指标（各指标最大值）:", ideal)
print()

# 3. 灰色关联分析
companies = df['公司名称'].tolist()
companies_data = df[metrics].values

gra = GreyRelationalAnalysis(rho=0.5)
gra.fit(ideal, companies_data,
        preprocess_method='initial',
        sequence_names=companies)

degrees = gra.get_correlation_degrees()
ranking = gra.get_ranking()

print("=" * 50)
print("灰色关联分析结果")
print("=" * 50)
print(f"{'公司':<10}{'关联度':<12}{'排名'}")
print("-" * 35)
for rank, company, degree in ranking:
    print(f"{company:<10}{degree:<12.4f}{rank}")
print()

# 4. 各指标影响分析
metric_importance = {}
for i, metric in enumerate(metrics):
    values = df[metric].values
    ideal_val = ideal[i]
    mad = np.mean(np.abs(values - ideal_val))
    metric_importance[metric] = 1.0 / (mad + 1e-10)

max_imp = max(metric_importance.values())
for k in metric_importance:
    metric_importance[k] = metric_importance[k] / max_imp

print("=" * 50)
print("各指标对公司整体评估的影响分析")
print("=" * 50)
print(f"{'指标':<15}{'影响权重':<12}{'说明'}")
print("-" * 50)
for metric, weight in sorted(metric_importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{metric:<15}{weight:<12.4f}区分度{'高' if weight > 0.7 else '中' if weight > 0.4 else '低'}")
print()

# 5. 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# 图1：关联度柱状图
ax1 = axes[0]
companies_sorted = [r[1] for r in ranking]
degrees_sorted = [r[2] for r in ranking]
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(companies_sorted)))[::-1]
bars = ax1.bar(companies_sorted, degrees_sorted, color=colors, edgecolor='black')
ax1.set_ylim(0, 1)
ax1.set_ylabel('灰色关联度 γ', fontsize=11)
ax1.set_title('各公司与理想公司的关联度', fontsize=12, fontweight='bold')
for bar, val in zip(bars, degrees_sorted):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02,
             f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# 图2：雷达图
ax2 = plt.subplot(1, 3, 2, projection='polar')
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]
for idx, company in enumerate(companies):
    values = df.loc[idx, metrics].values.tolist()
    values += values[:1]
    ax2.plot(angles, values, 'o-', linewidth=1.5, label=company, alpha=0.8)
    ax2.fill(angles, values, alpha=0.05)
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(metrics, fontsize=9)
ax2.set_ylim(0, 100)
ax2.set_title('各公司指标雷达图', fontsize=12, fontweight='bold', pad=15)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
ax2.grid(True)

# 图3：指标权重饼图
ax3 = axes[2]
metric_names = list(metric_importance.keys())
metric_weights = list(metric_importance.values())
explode = [0.05 if w == max(metric_weights) else 0 for w in metric_weights]
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(metric_names)))
wedges, texts, autotexts = ax3.pie(metric_weights, labels=metric_names, autopct='%1.1f%%',
                                    colors=colors_pie, explode=explode, startangle=90)
ax3.set_title('各指标影响权重', fontsize=12, fontweight='bold')
for text in texts:
    text.set_fontsize(9)
for autotext in autotexts:
    autotext.set_fontsize(9)
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.savefig('company_gra_analysis.png', dpi=150, bbox_inches='tight')
plt.show()