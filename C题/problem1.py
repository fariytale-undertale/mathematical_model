# -*- coding: utf-8 -*-
"""
2025 CUMCM C题 问题1
胎儿Y染色体浓度 与 孕周数 / BMI 等指标的相关特性分析、关系建模与显著性检验

数据说明:
- 附件.xlsx 的"男胎检测数据"共 1082 条记录, 来自 267 位孕妇(260 位有多次采血, 最多 8 次)
- 数据为纵向追踪: 同一孕妇在不同孕周多次采血 -> 需用混合效应模型处理重复测量
- 目标变量: Y染色体浓度(列V), 核心自变量: 检测孕周(列J), BMI(列K), 以及年龄/身高/体重
"""
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无界面后端, 直接保存图片
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# ---------------------------------------------------------------
# 0. 中文绘图字体配置 (CLAUDE.md 约定)
# ---------------------------------------------------------------
def setup_chinese_font():
    for font_path in fm.findSystemFonts():
        if any(name in font_path for name in ['SimHei', 'simhei', 'msyh', 'YaHei', 'SimSun', 'simsun']):
            try:
                fm.fontManager.addfont(font_path)
            except Exception:
                pass
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

setup_chinese_font()

import os
BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, "output", "figures")
RES_DIR = os.path.join(BASE, "output")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# ---------------------------------------------------------------
# 1. 数据加载与预处理
# ---------------------------------------------------------------
def week_to_num(s):
    """孕周字符串转连续数值: '11w+6' / '16W+1' -> 11.857 / 16.143"""
    s = str(s).strip().lower()
    m = re.match(r'(\d+)\s*w\s*\+?\s*(\d*)', s)
    if not m:
        return np.nan
    w = int(m.group(1))
    d = int(m.group(2)) if m.group(2) else 0
    return w + d / 7.0

def load_male_data():
    wb = pd.ExcelFile(os.path.join(BASE, "附件.xlsx"))
    df = wb.parse("男胎检测数据")
    # 重命名关键列(按附录说明)
    df = df.rename(columns={
        df.columns[1]: "code",      # 孕妇代码
        df.columns[2]: "age",       # 年龄
        df.columns[3]: "height",    # 身高
        df.columns[4]: "weight",    # 体重
        df.columns[9]: "week_str",  # 检测孕周(原始字符串)
        df.columns[10]: "BMI",      # BMI
        df.columns[21]: "Y_conc",   # Y染色体浓度
    })
    df["week"] = df["week_str"].apply(week_to_num)
    df["lnY"] = np.log(df["Y_conc"])  # 对数变换: 改善正偏态
    # 中心化, 避免二次项/交互项共线
    df["week_c"] = df["week"] - df["week"].mean()
    df["BMI_c"] = df["BMI"] - df["BMI"].mean()
    return df

df = load_male_data()
print("=" * 70)
print("问题1: Y染色体浓度与孕周、BMI的相关特性与关系模型")
print("=" * 70)
print(f"记录数: {len(df)},  唯一孕妇数: {df['code'].nunique()}")
print(f"多次采血孕妇数: {(df['code'].value_counts() > 1).sum()}")

# ---------------------------------------------------------------
# 2. 描述性统计
# ---------------------------------------------------------------
desc = df[["week", "BMI", "Y_conc", "age", "height", "weight"]].describe().T
desc = desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
desc.columns = ["样本量", "均值", "标准差", "最小值", "25%分位", "中位数", "75%分位", "最大值"]
print("\n[表1] 关键变量描述性统计")
print(desc.round(4).to_string())
desc.to_csv(os.path.join(RES_DIR, "表1_描述性统计.csv"), encoding="utf-8-sig")

# ---------------------------------------------------------------
# 3. 相关性分析
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("3. 相关性分析")
print("=" * 70)

vars_corr = df[["Y_conc", "week", "BMI", "age", "height", "weight"]]
vars_corr.columns = ["Y浓度", "孕周", "BMI", "年龄", "身高", "体重"]

# 3.1 Pearson 相关矩阵
pearson = vars_corr.corr(method="pearson")
print("\n[表2a] Pearson 相关矩阵")
print(pearson.round(3).to_string())

# 3.2 Spearman 相关矩阵(对非线性/异常值稳健)
spearman = vars_corr.corr(method="spearman")
print("\n[表2b] Spearman 相关矩阵")
print(spearman.round(3).to_string())

# 3.3 偏相关: 控制BMI后 孕周~Y浓度; 控制孕周后 BMI~Y浓度
def partial_corr(x, y, z):
    """控制变量 z 后, x 与 y 的偏相关系数"""
    r_xy = np.corrcoef(x, y)[0, 1]
    r_xz = np.corrcoef(x, z)[0, 1]
    r_yz = np.corrcoef(y, z)[0, 1]
    return (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))

pcorr_week = partial_corr(df["week"], df["Y_conc"], df["BMI"])
pcorr_bmi = partial_corr(df["BMI"], df["Y_conc"], df["week"])
# 偏相关显著性(t检验, df=n-3)
n = len(df)
def pcorr_pval(r, n):
    if abs(r) >= 1:
        return 0.0
    t = r * np.sqrt((n - 3) / (1 - r**2))
    return 2 * (1 - stats.t.cdf(abs(t), df=n - 3))

print("\n[表2c] 偏相关分析 (消除混杂)")
print(f"控制BMI后, 孕周~Y浓度 偏相关 r = {pcorr_week:.4f}  (p = {pcorr_pval(pcorr_week, n):.3e})")
print(f"控制孕周后, BMI~Y浓度  偏相关 r = {pcorr_bmi:.4f}  (p = {pcorr_pval(pcorr_bmi, n):.3e})")

# 4. 关系模型
print("\n" + "=" * 70)
print("4. 关系模型 (因变量 lnY = ln(Y浓度))")
print("=" * 70)

# 构建模型对比表
model_results = []

def fit_ols(formula, data, name):
    model = smf.ols(formula, data=data).fit()
    model_results.append((name, model))
    return model

# 模型1: 仅孕周
m1 = fit_ols("lnY ~ week_c", df, "M1: 仅孕周")
# 模型2: 孕周 + BMI
m2 = fit_ols("lnY ~ week_c + BMI_c", df, "M2: 孕周 + BMI")
# 模型3: 孕周 + BMI + 孕周二次项 (非线性)
m3 = fit_ols("lnY ~ week_c + BMI_c + I(week_c**2)", df, "M3: +孕周二次项")
# 模型4: 孕周 + BMI + 交互项
m4 = fit_ols("lnY ~ week_c + BMI_c + week_c:BMI_c", df, "M4: +交互项")
# 模型5: 全模型(年龄/身高/体重)
m5 = fit_ols("lnY ~ week_c + BMI_c + age + height + weight", df, "M5: 全模型")

print("\n[表3] 模型比较 (AIC/BIC/R²)")
comp = pd.DataFrame({
    "模型": [name for name, _ in model_results],
    "R²": [m.rsquared for _, m in model_results],
    "调整R²": [m.rsquared_adj for _, m in model_results],
    "AIC": [m.aic for _, m in model_results],
    "BIC": [m.bic for _, m in model_results],
    "对数似然": [m.llf for _, m in model_results],
})
print(comp.round(4).to_string(index=False))
comp.to_csv(os.path.join(RES_DIR, "表3_模型比较.csv"), encoding="utf-8-sig", index=False)

# 4.1 主模型 M2 的详细显著性检验
print("\n[表4] 模型M2 (lnY ~ week + BMI) 系数显著性检验")
print(m2.summary().tables[1])

# 4.2 嵌套模型 F 检验 (M2 是否显著优于仅截距; M3/M4 是否显著优于 M2)
print("\n[表5] 嵌套模型 F 检验")
from statsmodels.stats.anova import anova_lm
print("M2 vs 仅截距:")
m0 = smf.ols("lnY ~ 1", df).fit()
print(anova_lm(m0, m2).round(4).to_string())
print("\nM3(+二次项) vs M2:")
print(anova_lm(m2, m3).round(4).to_string())
print("\nM4(+交互项) vs M2:")
print(anova_lm(m2, m4).round(4).to_string())
print("\nM5(+年龄/身高/体重) vs M2:")
print(anova_lm(m2, m5).round(4).to_string())

# 4.3 混合效应模型(随机截距, 处理同一孕妇重复测量)
print("\n[表6] 混合效应模型 (随机截距, 正确处理纵向重复测量)")
try:
    md = smf.mixedlm("lnY ~ week_c + BMI_c", df, groups=df["code"])
    mdf = md.fit(reml=True)
    print(mdf.summary())
    # 组内方差 vs 组间方差
    icc = mdf.cov_re.iloc[0, 0] / (mdf.cov_re.iloc[0, 0] + mdf.scale)
    print(f"\n个体间方差占比 (ICC) = {icc:.4f}  -> 说明 {icc*100:.1f}% 的总变异来自孕妇个体差异")
except Exception as e:
    print("混合效应模型拟合失败:", e)

# 4.4 敏感性: 每孕妇取均值后重跑 M2
print("\n[表7] 敏感性分析: 每孕妇取均值后 M2 系数")
df_mean = df.groupby("code", as_index=False).agg(
    lnY=("lnY", "mean"), week_c=("week_c", "mean"),
    BMI_c=("BMI_c", "mean"), week=("week", "mean"), BMI=("BMI", "mean"))
m2_mean = smf.ols("lnY ~ week_c + BMI_c", df_mean).fit()
print(m2_mean.params.round(4).to_string())
print(f"(n={len(df_mean)}位孕妇) 调整R² = {m2_mean.rsquared_adj:.4f}")

# ---------------------------------------------------------------
# 5. 可视化
# ---------------------------------------------------------------
print("\n生成图表到 output/figures/ ...")

# 图1: Y浓度 vs 孕周, 按BMI分档着色
fig, ax = plt.subplots(figsize=(8, 6))
bmi_bins = pd.cut(df["BMI"], bins=[0, 28, 32, 36, 50],
                  labels=["BMI<28", "28≤BMI<32", "32≤BMI<36", "BMI≥36"])
sc = ax.scatter(df["week"], df["Y_conc"], c=df["BMI"], cmap="viridis",
                alpha=0.6, s=20, edgecolors="none")
ax.axhline(0.04, color="red", ls="--", lw=1.5, label="达标阈值 4%")
ax.set_xlabel("检测孕周 (周)")
ax.set_ylabel("Y染色体浓度")
ax.set_title("Y染色体浓度 与 孕周 的散点分布 (颜色=BMI)")
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("BMI")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig1_Y浓度_孕周_散点.png"), dpi=300)
plt.close(fig)

# 图2: Y浓度 vs BMI, 按孕周分档着色
fig, ax = plt.subplots(figsize=(8, 6))
week_bins = pd.cut(df["week"], bins=[10, 14, 18, 22, 26],
                   labels=["10-14周", "14-18周", "18-22周", "22-26周"])
sc = ax.scatter(df["BMI"], df["Y_conc"], c=df["week"], cmap="plasma",
                alpha=0.6, s=20, edgecolors="none")
ax.axhline(0.04, color="red", ls="--", lw=1.5, label="达标阈值 4%")
ax.set_xlabel("BMI")
ax.set_ylabel("Y染色体浓度")
ax.set_title("Y染色体浓度 与 BMI 的散点分布 (颜色=孕周)")
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("孕周 (周)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig2_Y浓度_BMI_散点.png"), dpi=300)
plt.close(fig)

# 图3: 相关矩阵热力图
fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(pearson, dtype=bool))
sns.heatmap(pearson, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            mask=mask, square=True, ax=ax, cbar_kws={"label": "相关系数"})
ax.set_title("Pearson 相关矩阵")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig3_相关矩阵.png"), dpi=300)
plt.close(fig)

# 图4: 模型M2残差诊断
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
resid = m2.resid
fitted = m2.fittedvalues
# (a) 残差 vs 拟合值
axes[0].scatter(fitted, resid, alpha=0.5, s=12)
axes[0].axhline(0, color="red", ls="--", lw=1)
axes[0].set_xlabel("拟合值")
axes[0].set_ylabel("残差")
axes[0].set_title("残差 vs 拟合值")
# (b) QQ图
stats.probplot(resid, dist="norm", plot=axes[1])
axes[1].set_title("残差正态性 QQ 图")
# (c) 残差直方图
axes[2].hist(resid, bins=40, density=True, alpha=0.7, edgecolor="black")
x = np.linspace(resid.min(), resid.max(), 200)
axes[2].plot(x, stats.norm.pdf(x, resid.mean(), resid.std()), "r-", lw=2)
axes[2].set_xlabel("残差")
axes[2].set_title("残差分布直方图")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig4_残差诊断.png"), dpi=300)
plt.close(fig)

# 图5: 分BMI组, Y浓度~孕周的拟合趋势
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for i, label in enumerate(bmi_bins.cat.categories):
    sub = df[bmi_bins == label]
    ax.scatter(sub["week"], sub["Y_conc"], s=14, alpha=0.4,
               color=colors[i], label=label)
    # 该组 Y~week 的线性拟合
    if len(sub) > 10:
        b, a = np.polyfit(sub["week"], sub["Y_conc"], 1)
        xs = np.linspace(sub["week"].min(), sub["week"].max(), 50)
        ax.plot(xs, a + b * xs, color=colors[i], lw=2)
ax.axhline(0.04, color="red", ls="--", lw=1.5, label="达标阈值 4%")
ax.set_xlabel("检测孕周 (周)")
ax.set_ylabel("Y染色体浓度")
ax.set_title("不同BMI组的 Y浓度~孕周 趋势 (线性拟合)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig5_分BMI组趋势.png"), dpi=300)
plt.close(fig)

print("\n所有图表已保存至:", FIG_DIR)
print("\n" + "=" * 70)
print("问题1 完成")
print("=" * 70)
