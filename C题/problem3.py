# -*- coding: utf-8 -*-
"""
2025 CUMCM C题 问题3
综合身高/体重/年龄/检测误差/达标比例 的 BMI分组与最佳NIPT时点

与问题2的差异:
1. 达标时间模型引入 年龄(age) 与 身高(height) 等多因素(体重与BMI共线剔除)
2. 显式建模"达标比例" F(t,X)=P(Y>=4%), 最佳时点取达标比例=80%的最早时间
3. 检测误差: 真实(σ_u) vs 测得(σ_total) + 测量误差σ_e敏感性 + 聚类bootstrap
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from data_utils import setup_chinese_font, load_male_data, FIG_DIR, RES_DIR
import matplotlib.pyplot as plt

setup_chinese_font()
LN_004 = np.log(0.04)

df = load_male_data()
# 中心化年龄/身高
age_mean, height_mean = df["age"].mean(), df["height"].mean()
df["age_c"] = df["age"] - age_mean
df["height_c"] = df["height"] - height_mean

# ---------------------------------------------------------------
# 1. 变量选择 (ML估计的AIC/BIC)
# ---------------------------------------------------------------
print("=" * 70)
print("问题3: 多因素综合的最佳NIPT时点")
print("=" * 70)
print("\n[表1] 变量选择 (混合效应模型, ML估计)")
models = {
    "M1: week+BMI": "lnY ~ week_c + BMI_c",
    "M2: +age": "lnY ~ week_c + BMI_c + age_c",
    "M3: +age+height": "lnY ~ week_c + BMI_c + age_c + height_c",
    "M4: +age+height+weight": "lnY ~ week_c + BMI_c + age_c + height_c + weight",
}
for name, f in models.items():
    m = smf.mixedlm(f, df, groups=df["code"]).fit(reml=False)
    print(f"  {name:<22} AIC={m.aic:7.1f}  BIC={m.bic:7.1f}  -2logLik={-2*m.llf:7.1f}")

# ---------------------------------------------------------------
# 2. 最终多因素模型 (REML)
# ---------------------------------------------------------------
md = smf.mixedlm("lnY ~ week_c + BMI_c + age_c + height_c", df, groups=df["code"]).fit(reml=True)
beta = md.params
sig_u2 = md.cov_re.iloc[0, 0]
sig_e2 = md.scale
sig_u = np.sqrt(sig_u2)
sig_total = np.sqrt(sig_u2 + sig_e2)
week_mean = df["week"].mean()
bmi_mean = df["BMI"].mean()

print("\n[表2] 多因素模型系数显著性 (REML)")
print(f"  Intercept={beta['Intercept']:.4f} (p={md.pvalues['Intercept']:.2e})")
for k in ["week_c", "BMI_c", "age_c", "height_c"]:
    print(f"  {k:<9} coef={beta[k]:.4f}  p={md.pvalues[k]:.4f}")
print(f"  组间σ_u={sig_u:.4f}  测量σ_e={np.sqrt(sig_e2):.4f}  ICC={sig_u2/(sig_u2+sig_e2):.3f}")

# ---------------------------------------------------------------
# 3. 达标比例函数 (多因素)
# ---------------------------------------------------------------
def mu_lnY(week, bmi, age, height):
    return (beta["Intercept"] + beta["week_c"] * (week - week_mean) +
            beta["BMI_c"] * (bmi - bmi_mean) + beta["age_c"] * (age - age_mean) +
            beta["height_c"] * (height - height_mean))

def F_attain(week, bmi, age, height, sigma):
    return 1 - stats.norm.cdf((LN_004 - mu_lnY(week, bmi, age, height)) / sigma)

def attain_time(bmi, age, height, z_alpha, sigma):
    return week_mean + (LN_004 + z_alpha * sigma - beta["Intercept"] -
                        beta["BMI_c"] * (bmi - bmi_mean) - beta["age_c"] * (age - age_mean) -
                        beta["height_c"] * (height - height_mean)) / beta["week_c"]

Z_DICT = {"80%": 0.842, "90%": 1.282, "95%": 1.645}

# ---------------------------------------------------------------
# 4. BMI分组, 每组用组内中位年龄/身高
# ---------------------------------------------------------------
def assign_group(bmi):
    if bmi < 28: return "G1: BMI<28"
    elif bmi < 32: return "G2: 28~32"
    elif bmi < 36: return "G3: 32~36"
    else: return "G4: BMI>=36"

df["group"] = df["BMI"].apply(assign_group)
groups = ["G1: BMI<28", "G2: 28~32", "G3: 32~36", "G4: BMI>=36"]

print("\n[表3] 各BMI组典型特征 与 最佳NIPT时点 (多因素, 80%测得达标)")
print(f"{'分组':<14}{'n':>5}{'BMI中位':>8}{'年龄中位':>8}{'身高中位':>8}{'最佳时点':>9}{'达标比例':>9}{'风险':>7}")

rows = []
for g in groups:
    sub = df[df["group"] == g]
    bmi_med = sub["BMI"].median()
    age_med = sub["age"].median()
    ht_med = sub["height"].median()
    t_opt = attain_time(bmi_med, age_med, ht_med, 0.842, sig_total)
    F_opt = F_attain(t_opt, bmi_med, age_med, ht_med, sig_total)
    rl = "低" if t_opt <= 12 else ("高" if t_opt <= 27 else "极高")
    rows.append([g, len(sub), bmi_med, age_med, ht_med, t_opt, F_opt, rl])
    print(f"{g:<14}{len(sub):>5}{bmi_med:>8.1f}{age_med:>8.0f}{ht_med:>8.0f}{t_opt:>9.2f}{F_opt:>9.3f}{rl:>7}")

res = pd.DataFrame(rows, columns=["分组", "n", "BMI中位", "年龄中位", "身高中位", "最佳时点", "达标比例", "风险等级"])
res.to_csv(f"{RES_DIR}/表3_多因素最佳时点.csv", encoding="utf-8-sig", index=False)

# ---------------------------------------------------------------
# 5. 与问题2(只看BMI)对比
# ---------------------------------------------------------------
print("\n[表4] 问题2(仅BMI) vs 问题3(多因素) 最佳时点对比")
md2 = smf.mixedlm("lnY ~ week_c + BMI_c", df, groups=df["code"]).fit(reml=True)
b0_2, bw_2, bb_2 = md2.params["Intercept"], md2.params["week_c"], md2.params["BMI_c"]
su_2 = np.sqrt(md2.cov_re.iloc[0, 0] + md2.scale)
for g, bmi_med in zip(groups, [df[df['group']==g]['BMI'].median() for g in groups]):
    t2 = week_mean + (LN_004 + 0.842 * su_2 - b0_2 - bb_2 * (bmi_med - bmi_mean)) / bw_2
    sub = df[df["group"] == g]
    t3 = attain_time(bmi_med, sub["age"].median(), sub["height"].median(), 0.842, sig_total)
    print(f"  {g:<14} 仅BMI={t2:.2f}周  多因素={t3:.2f}周  差异={t3-t2:+.2f}周")

# ---------------------------------------------------------------
# 6. 测量误差 σ_e 敏感性
# ---------------------------------------------------------------
print("\n[表5] 测量误差 σ_e 敏感性 (G3组, BMI中位, 最佳时点变化)")
bmi_g3 = df[df["group"] == "G3: 32~36"]["BMI"].median()
age_g3 = df[df["group"] == "G3: 32~36"]["age"].median()
ht_g3 = df[df["group"] == "G3: 32~36"]["height"].median()
sig_e0 = np.sqrt(sig_e2)
for factor in [0.5, 0.75, 1.0, 1.25, 1.5]:
    se = sig_e0 * factor
    st = np.sqrt(sig_u2 + se**2)
    t = attain_time(bmi_g3, age_g3, ht_g3, 0.842, st)
    print(f"  σ_e={se:.3f} (×{factor:.2f}): 最佳时点={t:.2f}周")

# ---------------------------------------------------------------
# 7. 聚类 bootstrap: 时点不确定性
# ---------------------------------------------------------------
print("\n[表6] 聚类bootstrap 最佳时点不确定性 (B=200)")
rng = np.random.default_rng(42)
B = 200
codes = df["code"].unique()
opt_samples = {g: [] for g in groups}
for b in range(B):
    sampled = rng.choice(codes, size=len(codes), replace=True)
    sub = df[df["code"].isin(sampled)]
    try:
        m = smf.mixedlm("lnY ~ week_c + BMI_c + age_c + height_c", sub, groups=sub["code"]).fit(reml=True)
    except Exception:
        continue
    su = np.sqrt(m.cov_re.iloc[0, 0] + m.scale)
    wm = sub["week"].mean()
    bm_ = sub["BMI"].mean()
    am_ = sub["age_c"].mean() + age_mean
    hm_ = sub["height_c"].mean() + height_mean
    for g in groups:
        bmi_med = sub[sub["group"] == g]["BMI"].median()
        age_med = sub[sub["group"] == g]["age"].median()
        ht_med = sub[sub["group"] == g]["height"].median()
        t = wm + (LN_004 + 0.842 * su - m.params["Intercept"] -
                  m.params["BMI_c"] * (bmi_med - bm_) - m.params["age_c"] * (age_med - am_) -
                  m.params["height_c"] * (ht_med - hm_)) / m.params["week_c"]
        opt_samples[g].append(t)

print(f"{'分组':<14}{'时点均值':>9}{'标准差':>9}{'95%置信区间':>18}")
for g in groups:
    arr = np.array(opt_samples[g])
    lo, hi = np.percentile(arr, [2.5, 97.5])
    print(f"{g:<14}{arr.mean():>9.2f}{arr.std():>9.3f}   [{lo:.1f}, {hi:.1f}]")

# ---------------------------------------------------------------
# 8. 可视化
# ---------------------------------------------------------------
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# 图1: 多因素达标比例曲线
fig, ax = plt.subplots(figsize=(8.5, 6))
t_grid = np.linspace(9, 28, 400)
for g, c in zip(groups, colors):
    sub = df[df["group"] == g]
    bmi_med, age_med, ht_med = sub["BMI"].median(), sub["age"].median(), sub["height"].median()
    F = F_attain(t_grid, bmi_med, age_med, ht_med, sig_total)
    ax.plot(t_grid, F, color=c, lw=2, label=f"{g} (BMI≈{bmi_med:.0f})")
    t_opt = attain_time(bmi_med, age_med, ht_med, 0.842, sig_total)
    ax.axvline(t_opt, color=c, ls="--", lw=1, alpha=0.7)
    ax.plot(t_opt, 0.8, "o", color=c, ms=7)
ax.axhline(0.8, color="gray", ls=":", lw=1)
ax.text(28.2, 0.8, "80%达标", va="center", fontsize=9, color="gray")
ax.axvline(12, color="red", ls=":", lw=1, alpha=0.5)
ax.set_xlabel("检测孕周 (周)")
ax.set_ylabel("达标比例 P(Y浓度≥4%)")
ax.set_title("问题3: 多因素达标比例曲线 (虚线=最佳时点)")
ax.set_ylim(0, 1.1)
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig3-1_多因素达标比例.png", dpi=300)
plt.close(fig)

# 图2: 年龄/身高对达标时间的影响 (G3组, 敏感性)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sub3 = df[df["group"] == "G3: 32~36"]
bmi_med, ht_med = sub3["BMI"].median(), sub3["height"].median()
age_grid = np.linspace(24, 40, 100)
axes[0].plot(age_grid, attain_time(bmi_med, age_grid, ht_med, 0.842, sig_total), lw=2)
axes[0].set_xlabel("年龄 (岁)")
axes[0].set_ylabel("最佳时点 (周)")
axes[0].set_title("年龄对最佳时点的影响 (G3组)")
axes[0].grid(alpha=0.3)

age_med = sub3["age"].median()
ht_grid = np.linspace(150, 172, 100)
axes[1].plot(ht_grid, attain_time(bmi_med, age_med, ht_grid, 0.842, sig_total), lw=2)
axes[1].set_xlabel("身高 (cm)")
axes[1].set_ylabel("最佳时点 (周)")
axes[1].set_title("身高对最佳时点的影响 (G3组)")
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig3-2_年龄身高敏感性.png", dpi=300)
plt.close(fig)

# 图3: 最佳时点柱状(问题2 vs 问题3)
fig, ax = plt.subplots(figsize=(8.5, 5.5))
g_labels = [g.split(":")[0] for g in groups]
x = np.arange(len(groups))
width = 0.35
t2_list, t3_list = [], []
for g in groups:
    sub = df[df["group"] == g]
    bmi_med = sub["BMI"].median()
    t2_list.append(week_mean + (LN_004 + 0.842 * su_2 - b0_2 - bb_2 * (bmi_med - bmi_mean)) / bw_2)
    t3_list.append(attain_time(bmi_med, sub["age"].median(), sub["height"].median(), 0.842, sig_total))
ax.bar(x - width/2, t2_list, width, label="仅BMI(问题2)", color="#1f77b4", alpha=0.8)
ax.bar(x + width/2, t3_list, width, label="多因素(问题3)", color="#d62728", alpha=0.8)
ax.axhline(12, color="red", ls=":", lw=1, alpha=0.6)
ax.text(3.4, 12.3, "12周", fontsize=8, color="red")
ax.set_xticks(x)
ax.set_xticklabels(g_labels)
ax.set_ylabel("最佳NIPT时点 (周)")
ax.set_title("问题2 vs 问题3 最佳时点对比")
ax.legend()
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig3-3_问题2vs3对比.png", dpi=300)
plt.close(fig)

# 图4: bootstrap时点分布
fig, ax = plt.subplots(figsize=(8, 5.5))
data_box = [opt_samples[g] for g in groups]
bp = ax.boxplot(data_box, tick_labels=g_labels, patch_artist=True, showmeans=True)
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.5)
ax.set_xlabel("BMI分组")
ax.set_ylabel("最佳NIPT时点 (周)")
ax.set_title("问题3: 检测误差下最佳时点分布 (B=200)")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig3-4_时点误差分布.png", dpi=300)
plt.close(fig)

print("\n问题3 完成, 图表已保存至:", FIG_DIR)
