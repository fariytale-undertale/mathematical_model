# -*- coding: utf-8 -*-
"""
2025 CUMCM C题 问题4
女胎异常判定 (监督分类): 以AB列非整倍体为标签, 综合Z值/GC含量/读段数/X染色体/BMI等因素

数据: 605条记录(147位孕妇), 正常538 / 异常67(11%不平衡)
关键观察: 异常样本Z值未显著偏离(|Z|>3仅0~4.5%), 单特征AUC≈0.5
=> 需综合多特征 + 非线性模型 + 类别不平衡处理
"""
import numpy as np
import pandas as pd
from data_utils import setup_chinese_font, load_female_data, FIG_DIR, RES_DIR
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, f1_score, recall_score, precision_score,
                             confusion_matrix, roc_curve, precision_recall_curve,
                             average_precision_score)

setup_chinese_font()
rng = np.random.default_rng(42)

df = load_female_data()

# ---------------------------------------------------------------
# 1. 特征工程
# ---------------------------------------------------------------
df["Z13_abs"] = df["Z13"].abs()
df["Z18_abs"] = df["Z18"].abs()
df["Z21_abs"] = df["Z21"].abs()
df["ZX_abs"] = df["ZX"].abs()
df["Z_sum"] = df["Z13"].abs() + df["Z18"].abs() + df["Z21"].abs()  # 多染色体联合
df["GC_dev"] = (df["GC"] - 0.5).abs()          # GC偏离正常中值
df["X_conc_abs"] = df["X_conc"].abs()
df["dup_filter"] = df["dup_ratio"] + df["filter_ratio"]  # 低质量读段综合

feature_cols = [
    "Z13", "Z18", "Z21", "ZX", "Z13_abs", "Z18_abs", "Z21_abs", "ZX_abs", "Z_sum",
    "GC", "GC13", "GC18", "GC21", "GC_dev",
    "X_conc", "X_conc_abs",
    "reads_total", "map_ratio", "dup_ratio", "filter_ratio", "dup_filter",
    "BMI", "age", "height", "weight", "week",
]

X = df[feature_cols].copy()
y = df["label"].values
# 缺失填充(仅BMI缺1)
X = X.fillna(X.median())

print("=" * 70)
print("问题4: 女胎异常判定 (监督分类)")
print("=" * 70)
print(f"样本: n={len(df)}, 正常={int((y==0).sum())}, 异常={int((y==1).sum())}")
print(f"特征数: {len(feature_cols)}")

# ---------------------------------------------------------------
# 2. 基线: Z值阈值法 (|Z|>3)
# ---------------------------------------------------------------
print("\n[表1] 基线: Z值阈值法 (|Z|>3 判异常)")
z_thresh = 3.0
pred_z = ((df[["Z13", "Z18", "Z21"]].abs() > z_thresh).any(axis=1)).astype(int)
cm_z = confusion_matrix(y, pred_z)
print(f"  混淆矩阵 [[TN,FP],[FN,TP]] = {cm_z.tolist()}")
print(f"  敏感度(召回)={recall_score(y, pred_z):.3f}  特异度={cm_z[0,0]/(cm_z[0,0]+cm_z[0,1]):.3f}  F1={f1_score(y, pred_z):.3f}")

# ---------------------------------------------------------------
# 3. 分类模型 (5折交叉验证)
# ---------------------------------------------------------------
print("\n[表2] 各模型 5折交叉验证性能 (处理类别不平衡)")
models = {
    "逻辑回归": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
    ]),
    "随机森林": RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                       random_state=42, n_jobs=-1),
    "梯度提升": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                           random_state=42),
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
for name, model in models.items():
    aucs, f1s, recs, precs, specs = [], [], [], [], []
    for tr, te in skf.split(X, y):
        model.fit(X.iloc[tr], y[tr])
        p = model.predict_proba(X.iloc[te])[:, 1]
        pred = (p > 0.5).astype(int)
        aucs.append(roc_auc_score(y[te], p))
        f1s.append(f1_score(y[te], pred))
        recs.append(recall_score(y[te], pred))
        precs.append(precision_score(y[te], pred))
        cm = confusion_matrix(y[te], pred)
        specs.append(cm[0, 0] / (cm[0, 0] + cm[0, 1]))
    rows.append([name, np.mean(aucs), np.mean(f1s), np.mean(recs), np.mean(precs), np.mean(specs)])
    print(f"  {name:<6} AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}  "
          f"F1={np.mean(f1s):.3f}  敏感度={np.mean(recs):.3f}  精确率={np.mean(precs):.3f}  特异度={np.mean(specs):.3f}")

perf = pd.DataFrame(rows, columns=["模型", "AUC", "F1", "敏感度", "精确率", "特异度"])
perf.to_csv(f"{RES_DIR}/表4_分类性能.csv", encoding="utf-8-sig", index=False)

# ---------------------------------------------------------------
# 4. 最佳模型(随机森林)的详细结果
# ---------------------------------------------------------------
rf = models["随机森林"]
y_proba = cross_val_predict(rf, X, y, cv=skf, method="predict_proba")[:, 1]
y_pred = (y_proba > 0.5).astype(int)

print("\n[表3] 随机森林 (最佳阈值0.5) 混淆矩阵")
cm = confusion_matrix(y, y_pred)
print(f"  [[TN={cm[0,0]:3d}  FP={cm[0,1]:3d}]")
print(f"   [FN={cm[1,0]:3d}  TP={cm[1,1]:3d}]]")
print(f"  敏感度={recall_score(y, y_pred):.3f}  特异度={cm[0,0]/(cm[0,0]+cm[0,1]):.3f}  "
      f"F1={f1_score(y, y_pred):.3f}  AUC={roc_auc_score(y, y_proba):.3f}")

# 阈值扫描: 找最佳F1阈值
print("\n[表4] 阈值对判定性能的影响 (随机森林)")
best_th, best_f1 = 0.5, 0
print(f"{'阈值':>6}{'敏感度':>9}{'特异度':>9}{'精确率':>9}{'F1':>8}")
th_rows = []
for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
    pred = (y_proba > th).astype(int)
    cm_ = confusion_matrix(y, pred)
    rec = recall_score(y, pred)
    spec = cm_[0, 0] / (cm_[0, 0] + cm_[0, 1])
    prec = precision_score(y, pred)
    f1 = f1_score(y, pred)
    th_rows.append([th, rec, spec, prec, f1])
    if f1 > best_f1:
        best_f1, best_th = f1, th
    print(f"{th:>6.2f}{rec:>9.3f}{spec:>9.3f}{prec:>9.3f}{f1:>8.3f}")
print(f"  -> 最佳F1阈值 = {best_th} (F1={best_f1:.3f})")

# ---------------------------------------------------------------
# 5. 特征重要性
# ---------------------------------------------------------------
rf_full = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)
rf_full.fit(X, y)
imp = pd.DataFrame({"特征": feature_cols, "重要性": rf_full.feature_importances_}).sort_values("重要性", ascending=False)
print("\n[表5] 特征重要性 Top10")
print(imp.head(10).to_string(index=False))
imp.to_csv(f"{RES_DIR}/表5_特征重要性.csv", encoding="utf-8-sig", index=False)

# ---------------------------------------------------------------
# 6. 可视化
# ---------------------------------------------------------------
# ROC曲线
fig, ax = plt.subplots(figsize=(6.5, 5.5))
for name, model in models.items():
    p = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="随机猜测")
ax.set_xlabel("假阳性率 (1-特异度)")
ax.set_ylabel("真阳性率 (敏感度)")
ax.set_title("女胎异常判定 ROC 曲线")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig4-1_ROC曲线.png", dpi=300)
plt.close(fig)

# PR曲线
fig, ax = plt.subplots(figsize=(6.5, 5.5))
for name, model in models.items():
    p = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
    prec, rec, _ = precision_recall_curve(y, p)
    ax.plot(rec, prec, lw=2, label=f"{name} (AP={average_precision_score(y, p):.3f})")
ax.set_xlabel("召回率 (敏感度)")
ax.set_ylabel("精确率")
ax.set_title("女胎异常判定 PR 曲线 (不平衡数据)")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig4-2_PR曲线.png", dpi=300)
plt.close(fig)

# 特征重要性条形图
fig, ax = plt.subplots(figsize=(8, 5.5))
top = imp.head(12).iloc[::-1]
ax.barh(top["特征"], top["重要性"], color="#1f77b4", alpha=0.8, edgecolor="black")
ax.set_xlabel("特征重要性")
ax.set_title("随机森林特征重要性 Top12")
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig4-3_特征重要性.png", dpi=300)
plt.close(fig)

# Z值分布对比(正常vs异常)
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for i, f in enumerate(["Z13", "Z18", "Z21", "ZX"]):
    axes[i].hist(df[df["label"] == 0][f], bins=30, alpha=0.6, density=True, label="正常", color="#1f77b4")
    axes[i].hist(df[df["label"] == 1][f], bins=30, alpha=0.6, density=True, label="异常", color="#d62728")
    axes[i].set_xlabel(f)
    axes[i].set_title(f"{f} 分布")
    if i == 0:
        axes[i].legend()
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig4-4_Z值分布对比.png", dpi=300)
plt.close(fig)

print("\n问题4 完成, 图表已保存至:", FIG_DIR)
