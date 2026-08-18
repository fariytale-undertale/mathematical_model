# -*- coding: utf-8 -*-
"""
2025 CUMCM C题 问题2
男胎孕妇 BMI 合理分组 + 每组最佳 NIPT 时点 (潜在风险最小) + 检测误差影响

建模思路(修正版):
1. 混合效应模型 lnY ~ week + BMI + (1|孕妇) 估计固定效应与方差分量
2. 达标比例 F(t,BMI) = P(Y(t)>=4%), 是"该BMI组中达标个体的比例"
   - 真实达标(σ_u, 组间个体差异): 反映生物学上的真实达标比例
   - 测得达标(σ_total=sqrt(σ_u^2+σ_e^2)): 反映含测量误差后实际测得达标的比例
3. 达标时间 t*(BMI, 置信度): F(t*)=置信度 的最早孕周 -> 这是"最早可靠检测时点"
4. 最佳NIPT时点 = 80%测得达标时间(三角论证: 90%/95%过于保守致时点过晚, 50%过早)
5. 检测误差影响 = 真实达标时间 与 测得达标时间 的差异 + 聚类bootstrap时点分布
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from data_utils import setup_chinese_font, load_male_data, FIG_DIR, RES_DIR
import matplotlib.pyplot as plt

setup_chinese_font()
LN_004 = np.log(0.04)

# ---------------------------------------------------------------
# 1. 混合效应模型
# ---------------------------------------------------------------
df = load_male_data()
md = smf.mixedlm("lnY ~ week_c + BMI_c", df, groups=df["code"]).fit(reml=True)
beta0, beta_w, beta_b = md.params["Intercept"], md.params["week_c"], md.params["BMI_c"]
sig_u2 = md.cov_re.iloc[0, 0]
sig_e2 = md.scale
sig_u = np.sqrt(sig_u2)
sig_total = np.sqrt(sig_u2 + sig_e2)
week_mean, bmi_mean = df["week"].mean(), df["BMI"].mean()

print("=" * 70)
print("问题2: BMI分组与最佳NIPT时点 (风险最小)")
print("=" * 70)
print(f"固定效应: 截距={beta0:.4f}  week_c={beta_w:.4f}  BMI_c={beta_b:.4f}")
print(f"方差分量: 组间σ_u={sig_u:.4f}  测量σ_e={np.sqrt(sig_e2):.4f}  总σ={sig_total:.4f}")

# ---------------------------------------------------------------
# 2. 达标比例与达标时间
# ---------------------------------------------------------------
def mu_lnY(week, bmi):
    return beta0 + beta_w * (week - week_mean) + beta_b * (bmi - bmi_mean)

def F_attain(week, bmi, sigma):
    """达标比例 P(Y>=4%) = 1 - Phi((ln0.04 - mu)/sigma)"""
    return 1 - stats.norm.cdf((LN_004 - mu_lnY(week, bmi)) / sigma)

def attain_time(bmi, z_alpha, sigma):
    """达标时间: mu - z_alpha*sigma = ln0.04 的最早孕周"""
    return week_mean + (LN_004 + z_alpha * sigma - beta0 - beta_b * (bmi - bmi_mean)) / beta_w

Z_DICT = {"80%": 0.842, "90%": 1.282, "95%": 1.645}

def risk_level(t):
    """题设三级延误风险"""
    return "低风险" if t <= 12 else ("高风险" if t <= 27 else "极高风险")

# ---------------------------------------------------------------
# 3. BMI 分组
# ---------------------------------------------------------------
def assign_group(bmi):
    if bmi < 28:
        return "G1: BMI<28"
    elif bmi < 32:
        return "G2: 28~32"
    elif bmi < 36:
        return "G3: 32~36"
    else:
        return "G4: BMI>=36"

df["group"] = df["BMI"].apply(assign_group)
groups = ["G1: BMI<28", "G2: 28~32", "G3: 32~36", "G4: BMI>=36"]
group_bmi_center = [26.0, 30.0, 34.0, 38.0]

print("\n[表1] BMI分组 (临床肥胖分级, 样本量约束)")
for g in groups:
    sub = df[df["group"] == g]
    print(f"  {g}: n={len(sub)}, BMI=[{sub['BMI'].min():.1f},{sub['BMI'].max():.1f}]")

# ---------------------------------------------------------------
# 4. 达标时间(真实 vs 测得, 多置信度) 与 最佳时点
# ---------------------------------------------------------------
print("\n[表2] 各BMI组达标时间 (真实达标用σ_u, 测得达标用σ_total)")
print(f"{'分组':<14}{'代表BMI':>7}" + "".join([f"{'测得'+z:>9}" for z in Z_DICT]) +
      f"{'真实90%':>9}{'最佳时点':>9}{'时点风险':>9}")

rows = []
for g, bmi_c in zip(groups, group_bmi_center):
    t_meas = {z: attain_time(bmi_c, zz, sig_total) for z, zz in Z_DICT.items()}
    t_true90 = attain_time(bmi_c, 1.282, sig_u)
    t_opt = t_meas["80%"]  # 最佳时点 = 80%测得达标时间
    rl = risk_level(t_opt)
    rows.append([g, bmi_c, t_meas["80%"], t_meas["90%"], t_meas["95%"], t_true90, t_opt, rl])
    print(f"{g:<14}{bmi_c:>7.0f}" + "".join([f"{t_meas[z]:>9.1f}" for z in Z_DICT]) +
          f"{t_true90:>9.1f}{t_opt:>9.1f}{rl:>9}")

res = pd.DataFrame(rows, columns=["分组", "代表BMI", "测得80%", "测得90%", "测得95%", "真实90%", "最佳时点", "时点风险"])
res.to_csv(f"{RES_DIR}/表2_各组达标时间与最佳时点.csv", encoding="utf-8-sig", index=False)

# 检测误差的量化: 同一置信度下 测得达标时间 - 真实达标时间
print("\n[表3] 检测误差对达标时间的影响 (90%置信, 测得-真实)")
err_rows = []
for g, bmi_c in zip(groups, group_bmi_center):
    t_meas90 = attain_time(bmi_c, 1.282, sig_total)
    t_true90 = attain_time(bmi_c, 1.282, sig_u)
    diff = t_meas90 - t_true90
    err_rows.append([g, bmi_c, t_meas90, t_true90, diff])
    print(f"  {g:<14} 测得={t_meas90:.2f}周  真实={t_true90:.2f}周  误差推迟={diff:.2f}周")
mean_err = np.mean([r[4] for r in err_rows])
print(f"  平均: 检测误差使达标时间推迟约 {mean_err:.2f} 周")

# ---------------------------------------------------------------
# 5. 聚类 bootstrap: 达标时间与最佳时点的不确定性
# ---------------------------------------------------------------
print("\n[表4] 聚类bootstrap 最佳时点不确定性 (B=200)")
rng = np.random.default_rng(42)
B = 200
codes = df["code"].unique()
opt_samples = {g: [] for g in groups}
t_grid = np.arange(10, 30, 1 / 7)

for b in range(B):
    sampled = rng.choice(codes, size=len(codes), replace=True)
    sub = df[df["code"].isin(sampled)]
    try:
        m = smf.mixedlm("lnY ~ week_c + BMI_c", sub, groups=sub["code"]).fit(reml=True)
    except Exception:
        continue
    b0, bw, bb = m.params["Intercept"], m.params["week_c"], m.params["BMI_c"]
    su = np.sqrt(m.cov_re.iloc[0, 0] + m.scale)
    wm, bm = sub["week"].mean(), sub["BMI"].mean()
    for g, bmi_c in zip(groups, group_bmi_center):
        t = wm + (LN_004 + 0.842 * su - b0 - bb * (bmi_c - bm)) / bw
        opt_samples[g].append(t)

print(f"{'分组':<14}{'时点均值':>9}{'标准差':>9}{'95%置信区间':>18}")
for g in groups:
    arr = np.array(opt_samples[g])
    lo, hi = np.percentile(arr, [2.5, 97.5])
    print(f"{g:<14}{arr.mean():>9.2f}{arr.std():>9.3f}   [{lo:.1f}, {hi:.1f}]")

# ---------------------------------------------------------------
# 6. 可视化
# ---------------------------------------------------------------
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# 图1: 各BMI组测得达标比例曲线 + 最佳时点
fig, ax = plt.subplots(figsize=(8.5, 6))
t_grid_plot = np.linspace(9, 28, 400)
for g, bmi_c, c in zip(groups, group_bmi_center, colors):
    F = F_attain(t_grid_plot, bmi_c, sig_total)
    ax.plot(t_grid_plot, F, color=c, lw=2, label=f"{g} (BMI≈{bmi_c:.0f})")
    t_opt = attain_time(bmi_c, 0.842, sig_total)
    ax.axvline(t_opt, color=c, ls="--", lw=1, alpha=0.7)
    ax.plot(t_opt, F_attain(t_opt, bmi_c, sig_total), "o", color=c, ms=7)
ax.axhline(0.8, color="gray", ls=":", lw=1)
ax.text(28.2, 0.8, "80%", va="center", fontsize=9, color="gray")
ax.axvline(12, color="red", ls=":", lw=1, alpha=0.5)
ax.text(12.1, 1.02, "12周", fontsize=8, color="red")
ax.set_xlabel("检测孕周 (周)")
ax.set_ylabel("测得达标比例 P(Y浓度≥4%)")
ax.set_title("各BMI组测得达标比例随孕周变化 (虚线=最佳时点)")
ax.set_ylim(0, 1.1)
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig2-1_达标比例曲线.png", dpi=300)
plt.close(fig)

# 图2: 达标时间 vs BMI (真实 vs 测得, 多置信度)
fig, ax = plt.subplots(figsize=(8, 5.5))
bmi_grid = np.linspace(22, 46, 200)
for z, lab, ls in [("80%", "测得80%", "-"), ("90%", "测得90%", "--"), ("95%", "测得95%", "-.")]:
    ax.plot(bmi_grid, attain_time(bmi_grid, Z_DICT[z], sig_total), ls=ls, lw=2, label=lab)
ax.plot(bmi_grid, attain_time(bmi_grid, 1.282, sig_u), ls=":", lw=2, color="gray", label="真实90%(无误差)")
ax.axhline(12, color="red", ls=":", lw=1, alpha=0.6)
ax.axhline(27, color="darkred", ls=":", lw=1, alpha=0.6)
ax.text(46.2, 12, "12周", va="center", fontsize=8, color="red")
ax.text(46.2, 27, "27周", va="center", fontsize=8, color="darkred")
ax.set_xlabel("BMI")
ax.set_ylabel("达标时间 (周)")
ax.set_title("达标时间随BMI变化 (真实vs测得)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig2-2_达标时间vsBMI.png", dpi=300)
plt.close(fig)

# 图3: 最佳时点 vs BMI组 (柱状) + 风险区间背景
fig, ax = plt.subplots(figsize=(8, 5.5))
t_opts = [attain_time(b, 0.842, sig_total) for b in group_bmi_center]
g_labels = [g.split(":")[0] for g in groups]
bars = ax.bar(g_labels, t_opts, color=colors, alpha=0.75, edgecolor="black")
ax.axhspan(0, 12, color="green", alpha=0.08, label="低风险(≤12周)")
ax.axhspan(12, 27, color="orange", alpha=0.08, label="高风险(13-27周)")
ax.axhspan(27, 30, color="red", alpha=0.12, label="极高风险(≥28周)")
for bar, t in zip(bars, t_opts):
    ax.text(bar.get_x() + bar.get_width() / 2, t + 0.3, f"{t:.1f}周",
            ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("最佳NIPT时点 (周)")
ax.set_xlabel("BMI分组")
ax.set_title("各BMI组最佳NIPT时点 (背景=风险等级)")
ax.set_ylim(0, 30)
ax.legend(loc="upper left")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig2-3_最佳时点柱状.png", dpi=300)
plt.close(fig)

# 图4: bootstrap 最佳时点分布
fig, ax = plt.subplots(figsize=(8, 5.5))
data_box = [opt_samples[g] for g in groups]
bp = ax.boxplot(data_box, tick_labels=g_labels, patch_artist=True, showmeans=True)
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.5)
ax.set_xlabel("BMI分组")
ax.set_ylabel("最佳NIPT时点 (周)")
ax.set_title("检测误差下最佳时点分布 (聚类bootstrap B=200)")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig2-4_时点误差分布.png", dpi=300)
plt.close(fig)

print("\n问题2 完成, 图表已保存至:", FIG_DIR)
