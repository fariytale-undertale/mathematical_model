# -*- coding: utf-8 -*-
"""问题4补充: 逻辑回归系数表 (可解释判定公式)
标准化特征 -> 逻辑回归(class_weight=balanced) -> 输出系数/截距/标准化公式
"""
import numpy as np
import pandas as pd
from data_utils import load_female_data
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = load_female_data()

# 与 problem4.py 一致的特征
df["Z13_abs"] = df["Z13"].abs()
df["Z18_abs"] = df["Z18"].abs()
df["Z21_abs"] = df["Z21"].abs()
df["ZX_abs"] = df["ZX"].abs()
df["Z_sum"] = df["Z13"].abs() + df["Z18"].abs() + df["Z21"].abs()
df["GC_dev"] = (df["GC"] - 0.5).abs()
df["X_conc_abs"] = df["X_conc"].abs()
df["dup_filter"] = df["dup_ratio"] + df["filter_ratio"]

feature_cols = [
    "Z13", "Z18", "Z21", "ZX", "Z13_abs", "Z18_abs", "Z21_abs", "ZX_abs", "Z_sum",
    "GC", "GC13", "GC18", "GC21", "GC_dev",
    "X_conc", "X_conc_abs",
    "reads_total", "map_ratio", "dup_ratio", "filter_ratio", "dup_filter",
    "BMI", "age", "height", "weight", "week",
]

X = df[feature_cols].copy().fillna(df[feature_cols].median())
y = df["label"].values

scaler = StandardScaler().fit(X)
Xz = scaler.transform(X)
lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
lr.fit(Xz, y)

coef = pd.DataFrame({
    "特征": feature_cols,
    "系数": lr.coef_[0],
    "均值": scaler.mean_,
    "标准差": scaler.scale_,
}).sort_values("系数", key=lambda s: s.abs(), ascending=False)

print("截距 beta0 =", lr.intercept_[0])
print("\n标准化公式: X' = (X - 均值)/标准差")
print(f"{'特征':<14}{'系数':>10}{'均值':>12}{'标准差':>12}")
for _, r in coef.iterrows():
    print(f"{r['特征']:<14}{r['系数']:>10.4f}{r['均值']:>12.4f}{r['标准差']:>12.4f}")

coef.to_csv("output/表6_逻辑回归系数.csv", encoding="utf-8-sig", index=False)
print("\n已保存 output/表6_逻辑回归系数.csv")
