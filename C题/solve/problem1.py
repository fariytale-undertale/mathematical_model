# -*- coding: utf-8 -*-
"""问题1：蔬菜各品类及单品销售量的分布规律及相互关系。"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from utils import setup_chinese_font, savefig, DATA_DIR, TAB_DIR, load_sales, load_items

setup_chinese_font()
import matplotlib as mpl
print("中文字体:", [f.name for f in mpl.font_manager.fontManager.ttflist if "SimHei" in f.name or "YaHei" in f.name][:3])

daily_cat = pd.read_pickle(f"{DATA_DIR}/daily_cat.pkl")
daily_item = pd.read_pickle(f"{DATA_DIR}/daily_item.pkl")
items = pd.read_csv(f"{DATA_DIR}/items.csv", dtype={"单品编码": str, "分类编码": str})
CATS = ["花叶类", "花菜类", "水生根茎类", "茄类", "辣椒类", "食用菌"]

# ============================================================
# 1.1 销量分布拟合（对数正态 vs 伽马 vs 正态，AIC 比较）
# ============================================================
print("\n===== 1.1 品类日销量分布拟合 =====")
aic_rows = []
for cat in CATS:
    x = daily_cat[daily_cat["分类名称"] == cat]["销量"].values
    x = x[x > 0]
    # 对数正态：对销量取 log 拟合正态
    logx = np.log(x)
    mu, sigma = logx.mean(), logx.std(ddof=1)
    lognorm_aic = 2 * 2 - 2 * np.sum(stats.norm.logpdf(logx, mu, sigma))  # 参数 mu,sigma
    # 伽马
    shape, loc, scale = stats.gamma.fit(x, floc=0)
    gamma_aic = 2 * 3 - 2 * np.sum(stats.gamma.logpdf(x, shape, loc, scale))
    # 正态
    mu_n, sigma_n = x.mean(), x.std(ddof=1)
    norm_aic = 2 * 2 - 2 * np.sum(stats.norm.logpdf(x, mu_n, sigma_n))
    aic_rows.append([cat, lognorm_aic, gamma_aic, norm_aic, mu, sigma, shape, scale])
aic_df = pd.DataFrame(aic_rows, columns=["品类", "对数正态AIC", "伽马AIC", "正态AIC", "log均值μ", "log标准差σ", "伽马shape", "伽马scale"])
aic_df["最优分布"] = aic_df[["对数正态AIC", "伽马AIC", "正态AIC"]].idxmin(axis=1).str.replace("AIC", "")
print(aic_df[["品类", "对数正态AIC", "伽马AIC", "正态AIC", "最优分布"]])
aic_df.to_csv(f"{TAB_DIR}/tab1_dist_fit.csv", index=False, encoding="utf-8-sig")

# ---- 图1：各品类日销量直方图 + 对数正态拟合 ----
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
axes = axes.ravel()
for i, cat in enumerate(CATS):
    ax = axes[i]
    x = daily_cat[daily_cat["分类名称"] == cat]["销量"].values
    x = x[x > 0]
    ax.hist(x, bins=30, density=True, alpha=0.6, color="steelblue", edgecolor="white")
    logx = np.log(x)
    mu, sigma = logx.mean(), logx.std(ddof=1)
    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, stats.lognorm.pdf(xs, sigma, scale=np.exp(mu)), "r-", lw=2, label="对数正态拟合")
    ax.set_title(cat)
    ax.legend(fontsize=7)
    ax.set_xlabel("日销量(kg)")
    ax.set_ylabel("密度")
fig.suptitle("各蔬菜品类日销量分布（对数正态拟合）", fontsize=14)
fig.tight_layout()
savefig(fig, "fig1_dist_fit.png")

# ============================================================
# 1.2 星期效应与季节效应
# ============================================================
print("\n===== 1.2 星期/季节效应 =====")
daily_cat["星期"] = daily_cat["销售日期"].dt.dayofweek
daily_cat["月份"] = daily_cat["销售日期"].dt.month
week_mean = daily_cat.groupby(["分类名称", "星期"])["销量"].mean().unstack()
week_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
week_mean.columns = week_names

fig, ax = plt.subplots(figsize=(11, 5))
for cat in CATS:
    ax.plot(range(7), week_mean.loc[cat].values, marker="o", label=cat)
ax.set_xticks(range(7)); ax.set_xticklabels(week_names)
ax.set_xlabel("星期"); ax.set_ylabel("平均日销量(kg)")
ax.set_title("各品类星期效应（一周平均销量）")
ax.legend()
ax.grid(alpha=0.3)
savefig(fig, "fig2_week_effect.png")

month_mean = daily_cat.groupby(["分类名称", "月份"])["销量"].mean().unstack()
fig, ax = plt.subplots(figsize=(11, 5))
for cat in CATS:
    ax.plot(month_mean.columns, month_mean.loc[cat].values, marker="o", label=cat)
ax.set_xlabel("月份"); ax.set_ylabel("平均日销量(kg)")
ax.set_title("各品类季节效应（月平均销量）")
ax.legend(); ax.grid(alpha=0.3)
savefig(fig, "fig3_month_effect.png")

# ============================================================
# 1.3 相关分析：Pearson vs Spearman（品类级）
# ============================================================
print("\n===== 1.3 品类销量相关 =====")
pivot = daily_cat.pivot_table(index="销售日期", columns="分类名称", values="销量", aggfunc="sum")
pivot = pivot[CATS].dropna()
pearson = pivot.corr(method="pearson")
spearman = pivot.corr(method="spearman")
pearson.to_csv(f"{TAB_DIR}/tab2_pearson.csv", encoding="utf-8-sig")
spearman.to_csv(f"{TAB_DIR}/tab2_spearman.csv", encoding="utf-8-sig")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, corr, title in [(axes[0], pearson, "Pearson 相关"), (axes[1], spearman, "Spearman 相关")]:
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(6)); ax.set_xticklabels(CATS, rotation=45, ha="right")
    ax.set_yticks(range(6)); ax.set_yticklabels(CATS)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.values[i, j]) > 0.6 else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle("各品类日销量相关矩阵（Pearson vs Spearman）", fontsize=13)
fig.tight_layout()
savefig(fig, "fig4_corr_matrix.png")

# ============================================================
# 1.4 关联规则挖掘（单品共现，按 日期+小时 构造购物篮）
# ============================================================
print("\n===== 1.4 单品关联规则 =====")
sales = load_sales()
sales["小时"] = sales["扫码销售时间"].astype(str).str[:2]
sales = sales[sales["单品编码"].isin(set(items["单品编码"]))]
# 篮子：日期 + 小时 内出现的单品集合
baskets = sales.groupby(["销售日期", "小时"])["单品编码"].apply(lambda s: set(s)).tolist()
n_baskets = len(baskets)
print("购物篮数量:", n_baskets)

# 单品出现频次
from collections import Counter
item_freq = Counter()
for b in baskets:
    for it in b:
        item_freq[it] += 1
item_name = dict(zip(items["单品编码"], items["单品名称"]))
item_cat = dict(zip(items["单品编码"], items["分类名称"]))

# 计算两两共现（仅保留出现频次足够高的单品，Top 80）
top_items = [it for it, c in item_freq.most_common(80)]
top_set = set(top_items)
pair_rules = []
for i in range(len(top_items)):
    for j in range(i + 1, len(top_items)):
        a, b = top_items[i], top_items[j]
        cnt_ab = sum(1 for bk in baskets if a in bk and b in bk)
        if cnt_ab < 5:
            continue
        cnt_a, cnt_b = item_freq[a], item_freq[b]
        sup = cnt_ab / n_baskets
        conf_ab = cnt_ab / cnt_a
        conf_ba = cnt_ab / cnt_b
        lift = sup / ((cnt_a / n_baskets) * (cnt_b / n_baskets))
        pair_rules.append([a, b, cnt_ab, sup, conf_ab, conf_ba, lift])
rules = pd.DataFrame(pair_rules, columns=["单品A", "单品B", "共现次数", "支持度", "置信度A→B", "置信度B→A", "提升度"])
rules["单品A名"] = rules["单品A"].map(item_name)
rules["单品B名"] = rules["单品B"].map(item_name)
rules["A类"] = rules["单品A"].map(item_cat)
rules["B类"] = rules["单品B"].map(item_cat)
rules = rules.sort_values("提升度", ascending=False)
rules_top = rules.head(20)
rules_top.to_csv(f"{TAB_DIR}/tab3_rules_top.csv", index=False, encoding="utf-8-sig")
print("提升度 Top 10 关联规则:")
print(rules_top[["单品A名", "单品B名", "A类", "B类", "支持度", "提升度"]].head(10).to_string(index=False))

print("\n问题1 完成。")
