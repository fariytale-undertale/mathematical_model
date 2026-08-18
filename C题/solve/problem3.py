# -*- coding: utf-8 -*-
"""问题3：单品级补货 + 定价（组合优化）。
双模型对比：
  基线：贪心固定选 Top 33（硬凑名额，含负利润单品）
  改进：单品数 N∈[27,33] 也纳入优化 + 品类覆盖约束 + 剔除负利润单品 + 局部搜索
补货：单品级报童模型（经验分布，残值 s=0），最小陈列量 2.5kg。
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import setup_chinese_font, savefig, DATA_DIR, TAB_DIR, load_sales

setup_chinese_font()

daily_item = pd.read_pickle(f"{DATA_DIR}/daily_item.pkl")
items = pd.read_csv(f"{DATA_DIR}/items.csv", dtype={"单品编码": str, "分类编码": str})
item_loss = pd.read_csv(f"{DATA_DIR}/item_loss.csv", dtype={"单品编码": str})
wholesale = pd.read_pickle(f"{DATA_DIR}/wholesale.pkl")
CATS = ["花叶类", "花菜类", "水生根茎类", "茄类", "辣椒类", "食用菌"]
hist_markup = {"水生根茎类": 0.490, "花叶类": 0.687, "花菜类": 0.538, "茄类": 0.575, "辣椒类": 0.595, "食用菌": 0.615}

# ============================================================
# 1. 候选单品（6/24-30 有销量）及参数
# ============================================================
win = daily_item[(daily_item["销售日期"] >= pd.Timestamp("2023-06-24")) &
                 (daily_item["销售日期"] <= pd.Timestamp("2023-06-30"))]
cand_codes = win["单品编码"].unique()

wholesale_recent = wholesale[(wholesale["日期"] >= pd.Timestamp("2023-06-01")) &
                             (wholesale["日期"] <= pd.Timestamp("2023-06-30"))].groupby("单品编码")["批发价格(元/千克)"].mean()

jun = daily_item[(daily_item["销售日期"] >= pd.Timestamp("2023-06-01")) &
                 (daily_item["销售日期"] <= pd.Timestamp("2023-06-30"))]
jun_pivot = jun.pivot_table(index="销售日期", columns="单品编码", values="销量", aggfunc="sum", fill_value=0)
jun_pivot = jun_pivot.reindex(pd.date_range("2023-06-01", "2023-06-30")).fillna(0)

cand = []
for code in cand_codes:
    cat = items[items["单品编码"] == code]["分类名称"].iloc[0]
    name = items[items["单品编码"] == code]["单品名称"].iloc[0]
    c = wholesale_recent.get(code, np.nan)
    if np.isnan(c):
        continue
    loss = item_loss[item_loss["单品编码"] == code]["损耗率(%)"].iloc[0] if len(item_loss[item_loss["单品编码"] == code]) > 0 else 0.0
    r = hist_markup[cat]
    p = c * (1 + r)
    daily_sales = jun_pivot[code].values if code in jun_pivot.columns else np.zeros(30)
    cand.append({"编码": code, "名称": name, "品类": cat, "批发价c": c, "损耗率": loss,
                 "售价p": p, "日均销量": daily_sales.mean(), "历史销量": daily_sales})
cand = pd.DataFrame(cand)
print(f"候选单品数: {len(cand)}")

# ============================================================
# 2. 单品报童补货 + 期望利润（经验分布，s=0）
# ============================================================
MIN_DISPLAY = 2.5

def item_newsvendor(row):
    c = row["批发价c"]; loss = row["损耗率"]; p = row["售价p"]
    c_eff = c / (1 - loss / 100) if loss < 100 else c
    cr = np.clip((p - c_eff) / p, 0.001, 0.999)
    sales = row["历史销量"]
    Q_eff = max(np.quantile(sales, cr), MIN_DISPLAY)
    Emin = np.mean(np.minimum(Q_eff, sales))
    profit = p * Emin - c_eff * Q_eff
    Q_order = Q_eff / (1 - loss / 100)
    return Q_eff, Q_order, profit

res = cand.apply(item_newsvendor, axis=1, result_type="expand")
cand["可售量Q"] = res[0]
cand["补货量Q"] = res[1]
cand["期望利润"] = res[2]
cand = cand.sort_values("期望利润", ascending=False).reset_index(drop=True)
n_pos = (cand["期望利润"] > 0).sum()
print(f"期望利润为正的单品数: {n_pos} / {len(cand)}")
print("\n期望利润 Top 15 单品:")
print(cand[["名称", "品类", "日均销量", "补货量Q", "售价p", "期望利润"]].head(15).round(2).to_string(index=False))

# ============================================================
# 3. 选品优化
# ============================================================
N_MIN, N_MAX = 27, 33

def total_profit(sel):
    return cand.loc[list(sel), "期望利润"].sum()

def coverage(sel):
    return cand.loc[list(sel), "品类"].value_counts().to_dict()

# ---- 基线：贪心固定 Top 33 ----
baseline_idx = set(cand.index[:N_MAX].tolist())
base_profit = total_profit(baseline_idx)
base_cov = coverage(baseline_idx)
print("\n基线(贪心固定Top33): 单品数=%d, 总期望利润=%.1f" % (len(baseline_idx), base_profit))

# ---- 改进：正利润优先 + 品类覆盖 + 单品数在[27,33]内优化 ----
def improved_select():
    selected = set(cand[cand["期望利润"] > 0].index)  # 所有正利润单品
    # 品类覆盖：缺失品类补选该品类利润最高的单品（即使负利润）
    covered = cand.loc[list(selected), "品类"].unique()
    for cat in CATS:
        if cat not in covered:
            cat_items = cand[cand["品类"] == cat]
            if len(cat_items) > 0:
                selected.add(cat_items.index[0])
    # 若仍 < 27，补充利润最高的负利润单品
    if len(selected) < N_MIN:
        rem = cand.loc[[i for i in cand.index if i not in selected]].sort_values("期望利润", ascending=False).index
        for i in rem:
            if len(selected) >= N_MIN:
                break
            selected.add(i)
    return selected

def local_search(selected):
    selected = set(selected)
    improved = True
    while improved:
        improved = False
        best_gain = 0; best_swap = None
        unselected = [i for i in cand.index if i not in selected]
        for i_out in list(selected):
            for i_in in unselected:
                new_sel = (selected - {i_out}) | {i_in}
                if not (N_MIN <= len(new_sel) <= N_MAX):
                    continue
                if not set(CATS).issubset(set(cand.loc[list(new_sel), "品类"].unique())):
                    continue
                gain = total_profit(new_sel) - total_profit(selected)
                if gain > best_gain:
                    best_gain = gain; best_swap = (i_out, i_in)
        if best_swap and best_gain > 1e-9:
            selected = (selected - {best_swap[0]}) | {best_swap[1]}
            improved = True
    return selected

imp_idx = local_search(improved_select())
imp_profit = total_profit(imp_idx)
imp_cov = coverage(imp_idx)
print("改进(正利润+品类覆盖+局部搜索): 单品数=%d, 总期望利润=%.1f" % (len(imp_idx), imp_profit))
print("  品类覆盖:", imp_cov)

compare = pd.DataFrame([
    {"模型": "基线(贪心固定Top33)", "单品数": len(baseline_idx), "总期望利润(元)": round(base_profit, 1),
     "正利润单品数": (cand.loc[list(baseline_idx), "期望利润"] > 0).sum(), "覆盖品类数": len(base_cov)},
    {"模型": "改进(正利润+品类覆盖+局部搜索)", "单品数": len(imp_idx), "总期望利润(元)": round(imp_profit, 1),
     "正利润单品数": (cand.loc[list(imp_idx), "期望利润"] > 0).sum(), "覆盖品类数": len(imp_cov)},
])
compare.to_csv(f"{TAB_DIR}/tab10_selection_compare.csv", index=False, encoding="utf-8-sig")
print("\n选品对比:")
print(compare.to_string(index=False))

# ============================================================
# 4. 最终7月1日补货量与定价表（改进模型选品）
# ============================================================
final = cand.loc[sorted(imp_idx)].sort_values(["品类", "期望利润"], ascending=[True, False])
final_out = final[["名称", "品类", "日均销量", "补货量Q", "售价p", "期望利润"]].copy()
final_out = final_out.rename(columns={"名称": "单品名称", "日均销量": "预测需求(kg)",
                                       "补货量Q": "补货量(kg)", "售价p": "售价(元/kg)", "期望利润": "期望利润(元)"})
final_out["补货量(kg)"] = final_out["补货量(kg)"].round(1)
final_out["预测需求(kg)"] = final_out["预测需求(kg)"].round(1)
final_out["售价(元/kg)"] = final_out["售价(元/kg)"].round(2)
final_out = final_out.reset_index(drop=True)
final_out.to_csv(f"{TAB_DIR}/tab11_final_items.csv", index=False, encoding="utf-8-sig")
print("\n7月1日 单品补货量与定价（改进模型，%d 个单品）:" % len(final_out))
print(final_out.to_string(index=False))
print("总补货量: %.1f kg, 总期望利润: %.1f 元" % (final["补货量Q"].sum(), final["期望利润"].sum()))

cat_summary = final.groupby("品类").agg(补货总量=("补货量Q", "sum"), 单品数=("期望利润", "size"),
                                        期望利润=("期望利润", "sum")).reset_index()
cat_summary.to_csv(f"{TAB_DIR}/tab12_cat_summary.csv", index=False, encoding="utf-8-sig")
print("\n品类补货汇总:")
print(cat_summary.round(1).to_string(index=False))

# ============================================================
# 5. 图表
# ============================================================
fig, ax = plt.subplots(figsize=(12, 4))
cand_sorted = cand.sort_values("期望利润", ascending=False).reset_index(drop=True)
colors = ["steelblue" if i in imp_idx else "lightgray" for i in cand_sorted.index]
ax.bar(range(len(cand_sorted)), cand_sorted["期望利润"], color=colors)
ax.axvline(len(imp_idx) - 0.5, color="red", ls="--", label=f"选品边界(N={len(imp_idx)})")
ax.set_xlabel("单品（按期望利润排序）"); ax.set_ylabel("期望利润(元)")
ax.set_title("单品期望利润排序与选品结果（蓝=入选，灰=未选）")
ax.legend(); ax.grid(axis="y", alpha=0.3)
savefig(fig, "fig8_item_selection.png")

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(cat_summary["品类"], cat_summary["单品数"], color="steelblue")
ax.set_xlabel("品类"); ax.set_ylabel("入选单品数")
ax.set_title("改进模型选品的品类分布")
ax.grid(axis="y", alpha=0.3)
savefig(fig, "fig9_selection_cat.png")

# ============================================================
# 6. 协同效应：在选品目标中引入关联规则的协同项
# ============================================================
print("\n===== 6. 协同效应 =====")
from collections import defaultdict
sales3 = load_sales()
sales3["小时"] = sales3["扫码销售时间"].astype(str).str[:2]
valid_codes = set(cand["编码"])
sales3 = sales3[sales3["单品编码"].isin(valid_codes)]
baskets3 = sales3.groupby(["销售日期", "小时"])["单品编码"].apply(set).tolist()
n_baskets3 = len(baskets3)

item_baskets = defaultdict(set)
item_freq = defaultdict(int)
for bi, b in enumerate(baskets3):
    for it in b:
        item_baskets[it].add(bi)
        item_freq[it] += 1

codes = list(cand["编码"])
profit_map = dict(zip(cand["编码"], cand["期望利润"]))
synergy_pairs = []  # (code_a, code_b, lift, min_profit)
for a in range(len(codes)):
    for b in range(a + 1, len(codes)):
        ca, cb = codes[a], codes[b]
        cnt_ab = len(item_baskets[ca] & item_baskets[cb])
        if cnt_ab < 5:
            continue
        pa = item_freq[ca] / n_baskets3
        pb = item_freq[cb] / n_baskets3
        lift = (cnt_ab / n_baskets3) / (pa * pb)
        if lift > 2.0:
            synergy_pairs.append((ca, cb, lift, min(profit_map[ca], profit_map[cb])))
print("强关联对(lift>2)数量:", len(synergy_pairs))

ALPHA = 0.30  # 协同系数：连带销售的利润转化率

def synergy_gain(sel_codes):
    sel = set(sel_codes)
    g = 0.0
    for ca, cb, lift, mp in synergy_pairs:
        if ca in sel and cb in sel:
            g += ALPHA * (lift - 1) * mp
    return g

def total_profit_with_syn(sel_codes):
    return sum(profit_map[c] for c in sel_codes) + synergy_gain(sel_codes)

def local_search_syn(selected):
    selected = set(selected)
    improved = True
    while improved:
        improved = False
        best_gain = 0; best_swap = None
        unselected = [c for c in codes if c not in selected]
        for c_out in list(selected):
            for c_in in unselected:
                new_sel = (selected - {c_out}) | {c_in}
                if not (N_MIN <= len(new_sel) <= N_MAX):
                    continue
                if not set(CATS).issubset(set(cand[cand["编码"].isin(new_sel)]["品类"].unique())):
                    continue
                gain = total_profit_with_syn(new_sel) - total_profit_with_syn(selected)
                if gain > best_gain:
                    best_gain = gain; best_swap = (c_out, c_in)
        if best_swap and best_gain > 1e-9:
            selected = (selected - {best_swap[0]}) | {best_swap[1]}
            improved = True
    return selected

imp_codes = set(cand.loc[list(imp_idx), "编码"])
syn_codes = local_search_syn(imp_codes)
syn_profit_base = sum(profit_map[c] for c in syn_codes)
syn_gain_val = synergy_gain(syn_codes)
syn_total = total_profit_with_syn(syn_codes)
print("无协同改进: 单品数=%d, 独立期望利润=%.1f" % (len(imp_idx), imp_profit))
print("有协同选品: 单品数=%d, 独立利润=%.1f, 协同增益=%.1f, 总目标=%.1f" % (
    len(syn_codes), syn_profit_base, syn_gain_val, syn_total))
added = syn_codes - imp_codes
removed = imp_codes - syn_codes
print("协同后新增单品:", [cand[cand['编码'] == c]['名称'].iloc[0] for c in added] if added else "无")
print("协同后移除单品:", [cand[cand['编码'] == c]['名称'].iloc[0] for c in removed] if removed else "无")

syn_compare = pd.DataFrame([
    {"模型": "独立报童+组合优化", "单品数": len(imp_idx), "独立利润(元)": round(imp_profit, 1), "协同增益(元)": 0.0, "总目标(元)": round(imp_profit, 1)},
    {"模型": "引入协同项后", "单品数": len(syn_codes), "独立利润(元)": round(syn_profit_base, 1), "协同增益(元)": round(syn_gain_val, 1), "总目标(元)": round(syn_total, 1)},
])
syn_compare.to_csv(f"{TAB_DIR}/tab13_synergy_compare.csv", index=False, encoding="utf-8-sig")
print("\n协同效应对比:")
print(syn_compare.to_string(index=False))

# ============================================================
# 7. 稳健性：批发价 ±10% 敏感性
# ============================================================
print("\n===== 7. 稳健性(批发价±10%) =====")
def re_evaluate(price_scale):
    rows = []
    for _, row in cand.iterrows():
        c = row["批发价c"] * price_scale
        loss = row["损耗率"]
        r = hist_markup[row["品类"]]
        p = c * (1 + r)
        c_eff = c / (1 - loss / 100) if loss < 100 else c
        cr = np.clip((p - c_eff) / p, 0.001, 0.999)
        sales_hist = row["历史销量"]
        Q_eff = max(np.quantile(sales_hist, cr), MIN_DISPLAY)
        Emin = np.mean(np.minimum(Q_eff, sales_hist))
        profit = p * Emin - c_eff * Q_eff
        rows.append({"编码": row["编码"], "期望利润": profit})
    return pd.DataFrame(rows).set_index("编码")["期望利润"]

robust_rows = []
for scale in [0.9, 1.0, 1.1]:
    prof = re_evaluate(scale)
    n_pos = (prof > 0).sum()
    sel_prof = prof[list(imp_codes)]
    n_neg = (sel_prof < 0).sum()
    total = sel_prof.sum()
    robust_rows.append([f"批发价×{scale:.1f}", n_pos, round(total, 1), n_neg])
    print(f"批发价×{scale:.1f}: 正利润单品数={n_pos}, 原28个入选总利润={total:.1f}, 其中转负={n_neg}个")
robust_df = pd.DataFrame(robust_rows, columns=["情景", "正利润单品数", "原入选28个总利润(元)", "转负单品数"])
robust_df.to_csv(f"{TAB_DIR}/tab14_robustness.csv", index=False, encoding="utf-8-sig")

print("\n问题3 完成。")
