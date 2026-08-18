# -*- coding: utf-8 -*-
"""P1 补充分析:
Part 1 问题一: 混合效应框架下的二次项显著性检验
Part 2 问题二: 置信水平 α 的敏感性扫描 (80% 阈值选择的鲁棒性)
Part 3 问题四: SMOTE 过采样 vs 类别权重 对比
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from data_utils import setup_chinese_font, load_male_data, load_female_data, FIG_DIR, RES_DIR
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix

setup_chinese_font()
rng = np.random.default_rng(42)
LN_004 = np.log(0.04)

print("=" * 70)
print("Part 1: 问题一 混合效应模型的二次项检验")
print("=" * 70)

df = load_male_data()
df["week_c2"] = df["week_c"] ** 2
df["BMI_c2"] = df["BMI_c"] ** 2

def fit_mm(formula, name):
    m = smf.mixedlm(formula, df, groups=df["code"]).fit(reml=False)  # ML 用于 AIC 比较
    print(f"  {name:<28} AIC={m.aic:7.2f}  -2logLik={-2*m.llf:7.2f}")
    return m

print("混合效应模型 (ML估计) 变量比较:")
m_base = fit_mm("lnY ~ week_c + BMI_c", "M0: week+BMI")
m_w2 = fit_mm("lnY ~ week_c + BMI_c + week_c2", "M1: +week^2")
m_b2 = fit_mm("lnY ~ week_c + BMI_c + BMI_c2", "M2: +BMI^2")
m_quad = fit_mm("lnY ~ week_c + BMI_c + week_c2 + BMI_c2", "M3: +week^2+BMI^2")

print("\n二次项系数的显著性 (M3):")
for k in ["week_c2", "BMI_c2"]:
    print(f"  {k:<8} coef={m_quad.params[k]:.5f}  p={m_quad.pvalues[k]:.3f}")

# 似然比检验 M0 vs M3
lr_stat = 2 * (m_quad.llf - m_base.llf)
p_lr = 1 - stats.chi2.cdf(lr_stat, df=2)
print(f"\n似然比检验 M0 vs M3: chi2={lr_stat:.3f}, df=2, p={p_lr:.3f}")
print(f"  -> {'二次项显著' if p_lr < 0.05 else '二次项不显著, 线性模型足够'}")

print("\n" + "=" * 70)
print("Part 2: 问题二 置信水平 α 的敏感性扫描")
print("=" * 70)

# 复用问题二模型 (week + BMI 混合效应)
md = smf.mixedlm("lnY ~ week_c + BMI_c", df, groups=df["code"]).fit(reml=True)
beta0, beta_w, beta_b = md.params["Intercept"], md.params["week_c"], md.params["BMI_c"]
sig_total = np.sqrt(md.cov_re.iloc[0, 0] + md.scale)
week_mean, bmi_mean = df["week"].mean(), df["BMI"].mean()

def attain_time(bmi, z_alpha):
    return week_mean + (LN_004 + z_alpha * sig_total - beta0 - beta_b * (bmi - bmi_mean)) / beta_w

groups = ["G1: BMI<28", "G2: 28~32", "G3: 32~36", "G4: BMI>=36"]
group_bmi = [26.0, 30.0, 34.0, 38.0]

alphas_show = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
print(f"{'置信水平α':>9}" + "".join([f"{'G'+str(i+1):>9}" for i in range(4)]))
rows = []
for a in alphas_show:
    za = stats.norm.ppf(a)
    ts = [attain_time(b, za) for b in group_bmi]
    rows.append([a] + ts)
    print(f"{a:>9.0%}" + "".join([f"{t:>9.2f}" for t in ts]))

res_a = pd.DataFrame(rows, columns=["置信水平α"] + groups)
res_a.to_csv(f"{RES_DIR}/表7_置信水平扫描.csv", encoding="utf-8-sig", index=False)

# 图: α 扫描
fig, ax = plt.subplots(figsize=(8, 5.5))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
a_grid = np.linspace(0.5, 0.97, 300)
for g, b, c in zip(groups, group_bmi, colors):
    ts = [attain_time(b, stats.norm.ppf(a)) for a in a_grid]
    ax.plot(a_grid * 100, ts, lw=2, color=c, label=f"{g} (BMI≈{b:.0f})")
ax.axvline(80, color="gray", ls="--", lw=1.5)
ax.text(80.5, ax.get_ylim()[0] + 0.3, "80%", fontsize=9, color="gray")
ax.axhline(12, color="red", ls=":", lw=1, alpha=0.6)
ax.text(96, 12.3, "12周", fontsize=8, color="red")
ax.set_xlabel("置信水平 α (%)")
ax.set_ylabel("最佳NIPT时点 (周)")
ax.set_title("最佳时点随置信水平 α 的变化 (80%为本文取值)")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig_supp_alpha扫描.png", dpi=300)
plt.close(fig)
print("已保存 fig_supp_alpha扫描.png")

print("\n" + "=" * 70)
print("Part 3: 问题四 SMOTE 过采样 vs 类别权重")
print("=" * 70)

df_f = load_female_data()
df_f["Z13_abs"] = df_f["Z13"].abs()
df_f["Z18_abs"] = df_f["Z18"].abs()
df_f["Z21_abs"] = df_f["Z21"].abs()
df_f["ZX_abs"] = df_f["ZX"].abs()
df_f["Z_sum"] = df_f["Z13"].abs() + df_f["Z18"].abs() + df_f["Z21"].abs()
df_f["GC_dev"] = (df_f["GC"] - 0.5).abs()
df_f["X_conc_abs"] = df_f["X_conc"].abs()
df_f["dup_filter"] = df_f["dup_ratio"] + df_f["filter_ratio"]

feature_cols = [
    "Z13", "Z18", "Z21", "ZX", "Z13_abs", "Z18_abs", "Z21_abs", "ZX_abs", "Z_sum",
    "GC", "GC13", "GC18", "GC21", "GC_dev", "X_conc", "X_conc_abs",
    "reads_total", "map_ratio", "dup_ratio", "filter_ratio", "dup_filter",
    "BMI", "age", "height", "weight", "week",
]
X = df_f[feature_cols].copy().fillna(df_f[feature_cols].median()).values
y = df_f["label"].values

def smote_minority(X_tr, y_tr, rng, k=5):
    """训练集内对少数类做 SMOTE"""
    X_min = X_tr[y_tr == 1]
    n_min, n_maj = len(X_min), int((y_tr == 0).sum())
    if n_min == 0 or n_maj - n_min <= 0:
        return X_tr, y_tr
    diff = n_maj - n_min
    k_eff = min(k, n_min - 1)
    if k_eff < 1:
        extra = X_min[rng.integers(n_min, size=diff)]
    else:
        nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_min)
        _, idx = nn.kneighbors(X_min)
        synth = []
        for _ in range(diff):
            i = int(rng.integers(n_min))
            j = int(rng.choice(idx[i][1:]))
            lam = rng.random()
            synth.append(X_min[i] + lam * (X_min[j] - X_min[i]))
        extra = np.array(synth)
    return np.vstack([X_tr, extra]), np.concatenate([y_tr, np.ones(len(extra))])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def cv_eval(make_clf, use_smote=False):
    aucs, recs, specs = [], [], []
    for tr, te in skf.split(X, y):
        X_tr, y_tr = X[tr], y[tr]
        if use_smote:
            X_tr, y_tr = smote_minority(X_tr, y_tr, rng)
        scaler = StandardScaler().fit(X_tr)
        clf = make_clf()
        clf.fit(scaler.transform(X_tr), y_tr)
        p = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        pred = (p > 0.5).astype(int)
        cm = confusion_matrix(y[te], pred)
        aucs.append(roc_auc_score(y[te], p))
        recs.append(recall_score(y[te], pred))
        specs.append(cm[0, 0] / (cm[0, 0] + cm[0, 1]))
    return np.mean(aucs), np.mean(recs), np.mean(specs)

lr_bw = lambda: LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
lr_plain = lambda: LogisticRegression(max_iter=2000, random_state=42)

auc_bw, rec_bw, spec_bw = cv_eval(lr_bw, use_smote=False)
auc_sm, rec_sm, spec_sm = cv_eval(lr_plain, use_smote=True)

print(f"{'方法':<18}{'AUC':>8}{'敏感度':>9}{'特异度':>9}")
print(f"{'类别权重平衡':<18}{auc_bw:>8.3f}{rec_bw:>9.3f}{spec_bw:>9.3f}")
print(f"{'SMOTE过采样':<18}{auc_sm:>8.3f}{rec_sm:>9.3f}{spec_sm:>9.3f}")
print(f"\n  -> SMOTE 相比类别权重: AUC {'提升' if auc_sm>auc_bw else '下降'} "
      f"{auc_sm-auc_bw:+.3f}, 敏感度 {'提升' if rec_sm>rec_bw else '下降'} {rec_sm-rec_bw:+.3f}")

res_sm = pd.DataFrame([
    ["类别权重平衡", auc_bw, rec_bw, spec_bw],
    ["SMOTE过采样", auc_sm, rec_sm, spec_sm],
], columns=["方法", "AUC", "敏感度", "特异度"])
res_sm.to_csv(f"{RES_DIR}/表8_SMOTE对比.csv", encoding="utf-8-sig", index=False)

print("\n全部补充分析完成")
