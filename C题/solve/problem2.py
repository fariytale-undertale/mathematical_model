# -*- coding: utf-8 -*-
"""问题2：品类级补货 + 定价。
双模型对比结构：
  需求预测：30天均值（基线稳健）；Holt-Winters 验证集无稳定提升 → 诚实报告
  补货决策：均值补货（基线） vs 报童最优补货（改进核心，考虑易逝损耗）
  定价决策：历史均衡加成率（面板回归证实数据无法识别负弹性）
残值 s=0（主假设）：蔬菜"当日未售、隔日无法再售"，未售出残值近似为0；
  敏感性分析 s∈{0,0.2p,0.4p,0.6p} 展示结论稳健性。
"""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from utils import setup_chinese_font, savefig, DATA_DIR, TAB_DIR, load_sales

setup_chinese_font()

daily_cat = pd.read_pickle(f"{DATA_DIR}/daily_cat.pkl")
daily_item = pd.read_pickle(f"{DATA_DIR}/daily_item.pkl")
cat_loss = pd.read_csv(f"{DATA_DIR}/cat_loss.csv", dtype={"小分类编码": str})
CATS = ["花叶类", "花菜类", "水生根茎类", "茄类", "辣椒类", "食用菌"]

# ============================================================
# 0. 促销折扣系数（说明：打折销售是主动促销，非滞销残值回收）
# ============================================================
sales = load_sales()
sales["销量(千克)"] = sales["销量(千克)"].astype(float)
sales["销售单价(元/千克)"] = sales["销售单价(元/千克)"].astype(float)
disc = sales[sales["是否打折销售"] == "是"]
normal = sales[sales["是否打折销售"] == "否"]
disc_price = (disc["销量(千克)"] * disc["销售单价(元/千克)"]).sum() / disc["销量(千克)"].sum()
normal_price = (normal["销量(千克)"] * normal["销售单价(元/千克)"]).sum() / normal["销量(千克)"].sum()
delta_promo = disc_price / normal_price
print(f"促销折扣系数 = {delta_promo:.3f}（主动促销，非残值）")

# ============================================================
# 1. 加权批发价 + 加成率分布
# ============================================================
daily_item["成本贡献"] = daily_item["批发价格(元/千克)"] * daily_item["正常销量"]
cat_cost = daily_item.groupby(["销售日期", "分类名称"]).apply(
    lambda g: g["成本贡献"].sum() / g["正常销量"].sum() if g["正常销量"].sum() > 0 else np.nan).rename("加权批发价")
daily_cat = daily_cat.merge(cat_cost.reset_index(), on=["销售日期", "分类名称"], how="left")
daily_cat["加成率"] = (daily_cat["平均售价"] / daily_cat["加权批发价"] - 1).clip(0, 2.0)
daily_cat["星期"] = daily_cat["销售日期"].dt.dayofweek
daily_cat["月份"] = daily_cat["销售日期"].dt.month

markup_stat = daily_cat.groupby("分类名称")["加成率"].agg(["mean", "std", "median"])
markup_stat["变异系数"] = markup_stat["std"] / markup_stat["mean"]
print("\n加成率分布:")
print(markup_stat.round(3))
markup_stat.round(3).to_csv(f"{TAB_DIR}/tab4_markup_stat.csv", encoding="utf-8-sig")

# ============================================================
# 2. 需求弹性：日度拟合(不稳定) vs 面板回归(控制FE/星期/月份)
# ============================================================
print("\n===== 弹性估计 =====")
d = daily_cat[(daily_cat["销量"] > 0) & (daily_cat["加成率"] > 0.02)].copy()
d["log销量"] = np.log(d["销量"])
d["log加成"] = np.log(1 + d["加成率"])
# 面板回归
X = pd.get_dummies(d[["分类名称", "星期", "月份"]], columns=["分类名称", "星期", "月份"], drop_first=True, dtype=float)
X["log加成"] = d["log加成"].values
X = sm.add_constant(X)
panel = sm.OLS(d["log销量"].values.astype(float), X.astype(float)).fit()
print(f"面板回归弹性 γ={panel.params['log加成']:+.3f}, p={panel.pvalues['log加成']:.3f}, "
      f"95%CI=[{panel.conf_int().loc['log加成',0]:+.3f}, {panel.conf_int().loc['log加成',1]:+.3f}], R²={panel.rsquared:.3f}")
elast_df = pd.DataFrame([{"模型": "面板回归(品类FE+星期+月份)", "弹性γ": round(panel.params["log加成"], 3),
                          "p值": round(panel.pvalues["log加成"], 3), "R²": round(panel.rsquared, 3)}])
elast_df.to_csv(f"{TAB_DIR}/tab4_elasticity.csv", index=False, encoding="utf-8-sig")

# ============================================================
# 3. 需求预测：30天均值（基线）；HW 对比（诚实报告）
# ============================================================
train_end = pd.Timestamp("2023-06-23")
pred_start = pd.Timestamp("2023-07-01")
pred_end = pd.Timestamp("2023-07-07")

def get_series(cat):
    return daily_cat[daily_cat["分类名称"] == cat].set_index("销售日期")["销量"].sort_index().asfreq("D").fillna(0)

sigma_log = {cat: np.log(get_series(cat)[get_series(cat) > 0]).std(ddof=1) for cat in CATS}

def forecast_mean(s, horizon):
    return np.repeat(s.iloc[-30:].mean(), horizon)

# 星期调整因子（各品类：星期 w 历史平均销量 / 全期平均销量）
dow_factor = {}
for cat in CATS:
    d = daily_cat[daily_cat["分类名称"] == cat]
    overall = d["销量"].mean()
    dow_factor[cat] = d.groupby("星期")["销量"].mean() / overall

# 未来7天星期序列（2023-07-01 是周六）
future_dow = [(pred_start + pd.Timedelta(days=i)).dayofweek for i in range(7)]

# 基线：30天均值（常数）；改进：30天均值 × 星期调整因子
pred_base = {cat: forecast_mean(get_series(cat), 7) for cat in CATS}
pred = {cat: pred_base[cat] * np.array([dow_factor[cat].get(w, 1.0) for w in future_dow]) for cat in CATS}
print("\n未来7天预测(30天均值×星期调整):")
for cat in CATS:
    print(f"  {cat}: {np.round(pred[cat], 1)}")

# 验证集：30天均值 vs 30天均值+星期调整 vs HW
from statsmodels.tsa.holtwinters import ExponentialSmoothing
val_start = pd.Timestamp("2023-06-24"); val_end = pd.Timestamp("2023-06-30")
val_rows = []
for cat in CATS:
    s = get_series(cat)
    tr = s[s.index <= train_end]
    va = s[(s.index >= val_start) & (s.index <= val_end)]
    actual = va.values
    m1 = forecast_mean(tr, 7)  # 30天均值（常数）
    val_dow = [d.dayofweek for d in va.index]
    m2 = m1 * np.array([dow_factor[cat].get(w, 1.0) for w in val_dow])  # 30天均值×星期调整
    try:
        m3 = np.maximum(ExponentialSmoothing(tr, seasonal="add", seasonal_periods=7, trend="add").fit().forecast(7).values, 0)
    except Exception:
        m3 = m1
    val_rows.append([cat, np.mean(np.abs(m1 - actual)), np.mean(np.abs(m2 - actual)), np.mean(np.abs(m3 - actual))])
val_df = pd.DataFrame(val_rows, columns=["品类", "30天均值MAE", "30天均值+星期调整MAE", "HoltWinters MAE"])
print("\n验证集 MAE 对比 (6/24-6/30):")
print(val_df.round(2).to_string(index=False))
val_df.to_csv(f"{TAB_DIR}/tab5_forecast_compare.csv", index=False, encoding="utf-8-sig")

# ============================================================
# 4. 报童模型（残值 s=0 主假设）
# ============================================================
recent_cost = daily_cat[daily_cat["销售日期"] >= pd.Timestamp("2023-06-01")].groupby("分类名称")["加权批发价"].mean()
hist_markup = daily_cat.groupby("分类名称")["加成率"].median()

def newsvendor_Q(mu_pred, sigma_log, cr):
    """对数正态需求下的报童最优可售量"""
    mu_log = np.log(mu_pred) - sigma_log ** 2 / 2
    return stats.lognorm.ppf(np.clip(cr, 0.001, 0.999), sigma_log, scale=np.exp(mu_log))

def newsvendor_profit(Q_eff, mu_log, sigma_log, p, c_eff, s):
    if Q_eff <= 0:
        return 0.0
    norm = stats.norm
    lnQ = np.log(Q_eff)
    Emin = np.exp(mu_log + sigma_log ** 2 / 2) * norm.cdf((lnQ - mu_log - sigma_log ** 2) / sigma_log) \
        + Q_eff * (1 - norm.cdf((lnQ - mu_log) / sigma_log))
    return p * Emin + s * (Q_eff - Emin) - c_eff * Q_eff

# 最终补货量与定价表（s=0）
final_rows = []
for cat in CATS:
    c = recent_cost[cat]
    loss_rate = float(cat_loss[cat_loss["小分类名称"] == cat]["平均损耗率(%)"].iloc[0])
    r = hist_markup[cat]
    p = c * (1 + r)
    c_eff = c / (1 - loss_rate / 100)
    cr = (p - c_eff) / p
    for d, mu_pred in enumerate(pred[cat]):
        Q_eff = newsvendor_Q(mu_pred, sigma_log[cat], cr)
        Q_order = Q_eff / (1 - loss_rate / 100)
        final_rows.append({
            "品类": cat, "日期": f"7月{d+1}日({['周一','周二','周三','周四','周五','周六','周日'][future_dow[d]]})",
            "预测需求(kg)": round(mu_pred, 1),
            "补货量(kg)": round(Q_order, 1), "售价(元/kg)": round(p, 2), "加成率": round(r, 3),
        })
final_df = pd.DataFrame(final_rows)
final_df.to_csv(f"{TAB_DIR}/tab7_final_replenish.csv", index=False, encoding="utf-8-sig")
print("\n未来一周补货量与定价（s=0, 报童模型, 历史均衡加成率）:")
print(final_df.to_string(index=False))

# ============================================================
# 5. 收益对比：均值补货 vs 报童补货（30天验证集 6月）
# ============================================================
def realized_profit(Q_order, D, c, p, loss_rate, s):
    Q_eff = Q_order * (1 - loss_rate / 100)
    sold = min(Q_eff, D)
    leftover = max(Q_eff - D, 0)
    return sold * p + leftover * s - Q_order * c

val_start = pd.Timestamp("2023-06-01"); val_end = pd.Timestamp("2023-06-30"); train_end = pd.Timestamp("2023-05-31")
profit_rows = []
for cat in CATS:
    s_series = get_series(cat)
    tr = s_series[s_series.index <= train_end]
    va = s_series[(s_series.index >= val_start) & (s_series.index <= val_end)]
    c = recent_cost[cat]
    loss_rate = float(cat_loss[cat_loss["小分类名称"] == cat]["平均损耗率(%)"].iloc[0])
    r = hist_markup[cat]; p = c * (1 + r)
    c_eff = c / (1 - loss_rate / 100); cr = (p - c_eff) / p
    sl = sigma_log[cat]
    base = imp = 0.0
    for D in va.values:
        mu = tr.iloc[-30:].mean()
        Q_base = mu / (1 - loss_rate / 100)
        base += realized_profit(Q_base, D, c, p, loss_rate, 0)
        Q_eff = newsvendor_Q(mu, sl, cr)
        Q_order = Q_eff / (1 - loss_rate / 100)
        imp += realized_profit(Q_order, D, c, p, loss_rate, 0)
    profit_rows.append([cat, base, imp, (imp - base) / abs(base) * 100])
profit_df = pd.DataFrame(profit_rows, columns=["品类", "均值补货利润(元)", "报童补货利润(元)", "提升(%)"])
print("\n30天验证集收益对比 (6月, s=0):")
print(profit_df.round(1).to_string(index=False))
print(f"合计: 均值补货 {profit_df['均值补货利润(元)'].sum():.0f} 元, "
      f"报童补货 {profit_df['报童补货利润(元)'].sum():.0f} 元, "
      f"总提升 {(profit_df['报童补货利润(元)'].sum()-profit_df['均值补货利润(元)'].sum())/abs(profit_df['均值补货利润(元)'].sum())*100:.1f}%")
profit_df.to_csv(f"{TAB_DIR}/tab8_profit_compare.csv", index=False, encoding="utf-8-sig")

# ============================================================
# 6. 残值敏感性：s ∈ {0, 0.2p, 0.4p, 0.6p}
# ============================================================
sens_rows = []
for s_frac in [0.0, 0.2, 0.4, 0.6]:
    tot_b = tot_i = 0.0
    for cat in CATS:
        s_series = get_series(cat)
        tr = s_series[s_series.index <= train_end]
        va = s_series[(s_series.index >= val_start) & (s_series.index <= val_end)]
        c = recent_cost[cat]; loss_rate = float(cat_loss[cat_loss["小分类名称"] == cat]["平均损耗率(%)"].iloc[0])
        r = hist_markup[cat]; p = c * (1 + r); c_eff = c / (1 - loss_rate / 100); sres = p * s_frac
        cr = (p - c_eff) / (p - sres) if (p - sres) > 1e-9 else 0.999
        sl = sigma_log[cat]
        for D in va.values:
            mu = tr.iloc[-30:].mean()
            Q_base = mu / (1 - loss_rate / 100)
            tot_b += realized_profit(Q_base, D, c, p, loss_rate, sres)
            Q_eff = newsvendor_Q(mu, sl, cr)
            Q_order = Q_eff / (1 - loss_rate / 100)
            tot_i += realized_profit(Q_order, D, c, p, loss_rate, sres)
    sens_rows.append([f"s={s_frac:.1f}p", tot_b, tot_i, (tot_i - tot_b) / abs(tot_b) * 100])
sens_df = pd.DataFrame(sens_rows, columns=["残值设定", "均值补货总利润(元)", "报童补货总利润(元)", "提升(%)"])
print("\n残值敏感性分析:")
print(sens_df.round(1).to_string(index=False))
sens_df.to_csv(f"{TAB_DIR}/tab9_salvage_sensitivity.csv", index=False, encoding="utf-8-sig")

# ============================================================
# 7. 图表
# ============================================================
fig, ax = plt.subplots(figsize=(12, 4))
cat = "花叶类"
s = get_series(cat)
ax.plot(s.index, s.values, lw=0.5, alpha=0.6, label="历史日销量")
fc = pred[cat]
dates = pd.date_range(pred_start, pred_end)
ax.plot(dates, fc, "r-o", lw=2, label="未来7天预测")
ax.axvline(train_end, color="gray", ls="--", alpha=0.5)
ax.set_xlim(pd.Timestamp("2023-04-01"), pred_end)
ax.set_title(f"{cat} 日销量时序与未来一周预测（30天均值）")
ax.set_xlabel("日期"); ax.set_ylabel("销量(kg)"); ax.legend()
savefig(fig, "fig5_forecast.png")

# 报童 vs 均值补货 收益柱状图
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(6); w = 0.38
ax.bar(x - w/2, profit_df["均值补货利润(元)"], w, label="均值补货(基线)")
ax.bar(x + w/2, profit_df["报童补货利润(元)"], w, label="报童补货(改进)")
ax.set_xticks(x); ax.set_xticklabels(CATS)
ax.set_ylabel("6月总利润(元)"); ax.set_title("均值补货 vs 报童补货 收益对比（30天验证集）")
ax.legend(); ax.grid(axis="y", alpha=0.3)
savefig(fig, "fig6_profit_compare.png")

# 残值敏感性曲线
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(sens_df["残值设定"], sens_df["提升(%)"], "o-", lw=2, color="steelblue")
ax.axhline(0, color="gray", ls="--", alpha=0.5)
ax.set_xlabel("残值设定 s"); ax.set_ylabel("报童相对均值补货的利润提升(%)")
ax.set_title("残值敏感性：报童模型价值随残值升高而降低")
ax.grid(alpha=0.3)
savefig(fig, "fig7_salvage_sensitivity.png")

print("\n问题2 完成。")
