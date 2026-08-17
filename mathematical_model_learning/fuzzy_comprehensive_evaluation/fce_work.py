"""
部门综合绩效模糊评价 —— 引用 fuzzy_comprehensive_evaluation 模块
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fuzzy_comprehensive_evaluation import (
    FuzzyComprehensiveEvaluation, FuzzyOperator, TriangularMF
)
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

# ========== 一、数据准备 ==========
raw = {
    "指标": ["任务完成时间","任务完成率","工作质量","沟通协作频率",
            "团队凝聚力","任务分配合理性","客户投诉率","客户满意度评分","客户重复购买率"],
    "维度": ["工作效率"]*3 + ["团队协作"]*3 + ["客户满意度"]*3,
    "类型": ["逆向","正向","正向","正向","正向","正向","逆向","正向","正向"],
    "部门A": [8, 0.95, 0.85, 7, 0.88, 0.90, 0.03, 0.92, 0.70],
    "部门B": [10, 0.90, 0.80, 5, 0.75, 0.85, 0.05, 0.85, 0.65],
    "部门C": [7, 0.98, 0.90, 6, 0.80, 0.80, 0.02, 0.95, 0.75],
}
df = pd.DataFrame(raw)
print("="*55)
print("原始数据")
print("="*55)
print(df.to_string(index=False))

# ========== 二、数据预处理（统一量纲到[0,1]） ==========
# 对每个指标按类型归一化：正向指标越大越好，逆向指标越小越好
def normalize(col, indicator_type):
    col = np.asarray(col, dtype=float)
    mn, mx = col.min(), col.max()
    if mx - mn < 1e-10: return np.ones_like(col)
    return (col - mn)/(mx - mn) if indicator_type == "正向" else (mx - col)/(mx - mn)

norm = np.zeros((9, 3))
for i in range(9):
    norm[i] = normalize([raw["部门A"][i], raw["部门B"][i], raw["部门C"][i]], raw["类型"][i])

df_norm = pd.DataFrame(norm, columns=["部门A","部门B","部门C"], index=raw["指标"])
print("\n" + "="*55)
print("归一化数据 [0,1]")
print("="*55)
print(df_norm.round(4).to_string())

# ========== 三、构建模糊关系矩阵（三角隶属度函数） ==========
comments = ["优秀", "良好", "一般", "较差"]

def membership(v):
    """归一化值→4个评语隶属度，使用有重叠的三角函数"""
    return np.array([
        TriangularMF(0.50, 0.85, 1.20)(v),   # 优秀 [0.5,1.0]峰值0.85
        TriangularMF(0.25, 0.55, 0.85)(v),    # 良好
        TriangularMF(0.00, 0.30, 0.60)(v),    # 一般
        TriangularMF(-0.30, 0.05, 0.35)(v),   # 较差
    ])

def build_R(idx):
    R = np.zeros((9, 4))
    for i in range(9):
        R[i] = membership(norm[i, idx])
    # 行归一化
    rs = R.sum(axis=1, keepdims=True)
    rs[rs==0] = 1
    return R / rs

R_A, R_B, R_C = build_R(0), build_R(1), build_R(2)

# ========== 四、权重设置 ==========
w_dim = np.array([0.40, 0.30, 0.30])        # 一级：三个维度
w_eff = np.array([0.30, 0.40, 0.30])        # 二级：工作效率
w_team = np.array([0.30, 0.40, 0.30])       # 二级：团队协作
w_sat = np.array([0.30, 0.40, 0.30])        # 二级：客户满意度

# ========== 五、多级模糊综合评价 ==========
def evaluate_dept(R, name):
    # 子系统1: 工作效率
    f1 = FuzzyComprehensiveEvaluation()
    f1.set_factors(["任务完成时间","任务完成率","工作质量"])
    f1.set_comments(comments)
    f1.set_weights(w_eff)
    f1.set_relation_matrix(R[0:3])
    r1 = f1.evaluate()
    
    # 子系统2: 团队协作
    f2 = FuzzyComprehensiveEvaluation()
    f2.set_factors(["沟通协作频率","团队凝聚力","任务分配合理性"])
    f2.set_comments(comments)
    f2.set_weights(w_team)
    f2.set_relation_matrix(R[3:6])
    r2 = f2.evaluate()
    
    # 子系统3: 客户满意度
    f3 = FuzzyComprehensiveEvaluation()
    f3.set_factors(["客户投诉率","客户满意度评分","客户重复购买率"])
    f3.set_comments(comments)
    f3.set_weights(w_sat)
    f3.set_relation_matrix(R[6:9])
    r3 = f3.evaluate()
    
    # 一级综合评价
    ft = FuzzyComprehensiveEvaluation()
    ft.set_factors(["工作效率","团队协作","客户满意度"])
    ft.set_comments(comments)
    ft.set_weights(w_dim)
    ft.set_relation_matrix(np.vstack([r1, r2, r3]))
    total = ft.evaluate()
    score = ft.compute_score([4, 3, 2, 1])
    best, val = ft.get_max_membership_comment()
    
    return {"name": name, "sub": {"工作效率": r1, "团队协作": r2, "客户满意度": r3},
            "total": total, "score": score, "best": best, "val": val}

resA = evaluate_dept(R_A, "部门A")
resB = evaluate_dept(R_B, "部门B")
resC = evaluate_dept(R_C, "部门C")

# ========== 六、结果输出 ==========
print("\n" + "="*55)
print("多级模糊综合评价结果")
print("="*55)
for r in [resA, resB, resC]:
    print(f"\n【{r['name']}】")
    for dim, vec in r["sub"].items():
        print(f"  {dim}: {vec.round(4)} → 得分: {np.dot(vec,[4,3,2,1]):.4f}")
    print(f"  综合: {r['total'].round(4)}  得分: {r['score']:.4f}  等级: {r['best']}")

# 排名
deps = sorted([resA, resB, resC], key=lambda x: x["score"], reverse=True)
print("\n" + "="*55)
print("综合排名")
print("="*55)
for i, d in enumerate(deps, 1):
    print(f"  第{i}名: {d['name']} (得分:{d['score']:.4f}, {d['best']})")

# ========== 七、可视化 ==========
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
deps_ord = [resA, resB, resC]
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
x = np.arange(4)
w = 0.25

# 图1: 综合评语隶属度对比
ax1 = axes[0]
for i, (r, c) in enumerate(zip(deps_ord, colors)):
    ax1.bar(x + i*w, r["total"], w, label=r["name"], color=c, edgecolor="white")
ax1.set_xticks(x + w); ax1.set_xticklabels(comments)
ax1.set_ylabel("隶属度"); ax1.set_title("各部门综合评语隶属度对比")
ax1.legend(); ax1.set_ylim(0, 0.7)
for s in ["top","right"]: ax1.spines[s].set_visible(False)

# 图2: 综合得分对比
ax2 = axes[1]
names = [d["name"] for d in deps_ord]
scores = [d["score"] for d in deps_ord]
bars = ax2.bar(names, scores, color=colors, edgecolor="white", width=0.5)
ax2.set_ylabel("综合得分"); ax2.set_title("各部门综合得分对比"); ax2.set_ylim(0, 4)
for bar, sc in zip(bars, scores):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f"{sc:.2f}",
             ha="center", va="bottom", fontsize=11, fontweight="bold")
for s in ["top","right"]: ax2.spines[s].set_visible(False)

# 图3: 各维度得分折线图
ax3 = axes[2]
dims = ["工作效率", "团队协作", "客户满意度"]
dim_sc = {d["name"]: [np.dot(d["sub"][dim],[4,3,2,1]) for dim in dims] for d in deps_ord}
xd = np.arange(3)
for i, (name, c) in enumerate(zip(["部门A","部门B","部门C"], colors)):
    ax3.plot(xd, dim_sc[name], "o-", color=c, label=name, linewidth=2, markersize=8)
ax3.set_xticks(xd); ax3.set_xticklabels(dims)
ax3.set_ylabel("维度得分"); ax3.set_title("各维度得分对比"); ax3.legend()
ax3.set_ylim(1, 4.5); ax3.grid(True, alpha=0.3)
for s in ["top","right"]: ax3.spines[s].set_visible(False)

plt.tight_layout()
plt.savefig("dept_evaluation_result.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n图表已保存")