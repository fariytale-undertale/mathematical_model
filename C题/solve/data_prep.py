# -*- coding: utf-8 -*-
"""数据清洗与预处理：生成 日×单品、日×品类 聚合表，供后续问题复用。"""
import pandas as pd
import numpy as np
from utils import BASE, DATA_DIR, load_items, load_sales, load_wholesale, load_loss

print("=" * 60)
print("读入附件...")
items = load_items()          # 251 单品 -> 品类
sales = load_sales()          # 87.8万行销售流水
wholesale = load_wholesale()  # 批发价格
item_loss, cat_loss = load_loss()

# ---------- 清洗销售流水 ----------
print("销售类型分布:", sales["销售类型"].value_counts().to_dict())
print("退货记录销量符号示例:")
print(sales[sales["销售类型"] == "退货"]["销量(千克)"].describe())

# 退货记录销量为负，直接求和即可得到净销量；销售类型=销售为正常销售
sales["销量(千克)"] = sales["销量(千克)"].astype(float)
sales["销售单价(元/千克)"] = sales["销售单价(元/千克)"].astype(float)

# 异常值清洗：负销量仅保留退货记录（销售记录销量应为正）
neg_sales = sales[(sales["销售类型"] == "销售") & (sales["销量(千克)"] < 0)]
print("销售记录中负销量条数:", len(neg_sales))
sales = sales[~((sales["销售类型"] == "销售") & (sales["销量(千克)"] < 0))]

# 单品编码需在附件1中（有效单品）
valid_items = set(items["单品编码"])
sales = sales[sales["单品编码"].isin(valid_items)]
wholesale = wholesale[wholesale["单品编码"].isin(valid_items)]
print("清洗后销售流水行数:", len(sales))

# ---------- 聚合：日 × 单品 ----------
# 净销量（销售正 + 退货负）
g = sales.groupby(["销售日期", "单品编码"]).agg(
    销量=("销量(千克)", "sum"),
    销售额=("销量(千克)", lambda s: (s * sales.loc[s.index, "销售单价(元/千克)"]).sum()),
    销售条数=("销量(千克)", "size"),
)
# 打折销量（打折销售记录，视为残值/降价销售）
disc = sales[sales["是否打折销售"] == "是"]
disc_g = disc.groupby(["销售日期", "单品编码"]).agg(
    打折销量=("销量(千克)", "sum"),
)
normal = sales[sales["是否打折销售"] == "否"]
normal_g = normal.groupby(["销售日期", "单品编码"]).agg(
    正常销量=("销量(千克)", "sum"),
    正常销售额=("销量(千克)", lambda s: (s * normal.loc[s.index, "销售单价(元/千克)"]).sum()),
)
daily_item = g.join(disc_g, how="left").join(normal_g, how="left")
daily_item["打折销量"] = daily_item["打折销量"].fillna(0)
daily_item["正常销量"] = daily_item["正常销量"].fillna(0)
daily_item["正常销售额"] = daily_item["正常销售额"].fillna(0)
# 加权平均售价（正常销售）
daily_item["平均售价"] = daily_item["正常销售额"] / daily_item["正常销量"].replace(0, np.nan)
daily_item = daily_item.reset_index()

# ---------- 合并品类、损耗率、批发价 ----------
item_map = items[["单品编码", "单品名称", "分类编码", "分类名称"]]
daily_item = daily_item.merge(item_map, on="单品编码", how="left")
daily_item = daily_item.merge(item_loss[["单品编码", "损耗率(%)"]], on="单品编码", how="left")
daily_item["损耗率(%)"] = daily_item["损耗率(%)"].fillna(0)

# 批发价：按 日期×单品 合并；缺批发价的日期用该单品全期均价填充
wholesale["日期"] = pd.to_datetime(wholesale["日期"])
daily_item = daily_item.merge(wholesale, left_on=["销售日期", "单品编码"], right_on=["日期", "单品编码"], how="left")
daily_item.drop(columns=["日期"], inplace=True)
item_mean_wp = wholesale.groupby("单品编码")["批发价格(元/千克)"].mean().rename("批发价填充")
daily_item = daily_item.merge(item_mean_wp, on="单品编码", how="left")
daily_item["批发价格(元/千克)"] = daily_item["批发价格(元/千克)"].fillna(daily_item["批发价填充"])
daily_item.drop(columns=["批发价填充"], inplace=True)
# 仅保留有销量且有批发价的有效行
daily_item = daily_item.dropna(subset=["批发价格(元/千克)"])

print("日×单品 聚合行数:", len(daily_item))
print("覆盖日期:", daily_item["销售日期"].min(), "->", daily_item["销售日期"].max())

# ---------- 聚合：日 × 品类 ----------
daily_cat = daily_item.groupby(["销售日期", "分类编码", "分类名称"]).agg(
    销量=("销量", "sum"),
    正常销量=("正常销量", "sum"),
    销售额=("销售额", "sum"),
    正常销售额=("正常销售额", "sum"),
    打折销量=("打折销量", "sum"),
    批发成本=("批发价格(元/千克)", "mean"),  # 品类内简单均值，作近似成本
).reset_index()
daily_cat["平均售价"] = daily_cat["正常销售额"] / daily_cat["正常销量"].replace(0, np.nan)

# 品类损耗率：用附件4小分类平均损耗率
daily_cat = daily_cat.merge(cat_loss[["小分类编码", "平均损耗率(%)"]], left_on="分类编码", right_on="小分类编码", how="left")
daily_cat["平均损耗率(%)"] = daily_cat["平均损耗率(%)"].fillna(0)

# ---------- 保存 ----------
items.to_csv(f"{DATA_DIR}/items.csv", index=False, encoding="utf-8-sig")
item_loss.to_csv(f"{DATA_DIR}/item_loss.csv", index=False, encoding="utf-8-sig")
cat_loss.to_csv(f"{DATA_DIR}/cat_loss.csv", index=False, encoding="utf-8-sig")
daily_item.to_pickle(f"{DATA_DIR}/daily_item.pkl")
daily_cat.to_pickle(f"{DATA_DIR}/daily_cat.pkl")
wholesale.to_pickle(f"{DATA_DIR}/wholesale.pkl")

print("=" * 60)
print("已保存中间结果到 solve/cache/")
print("品类日销量概况:")
print(daily_cat.groupby("分类名称")["销量"].agg(["count", "mean", "sum"]))
print("各品类损耗率:")
print(cat_loss)
