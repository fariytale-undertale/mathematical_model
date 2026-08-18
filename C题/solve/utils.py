# -*- coding: utf-8 -*-
"""公共工具：中文字体、数据加载、输出路径"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np

BASE = r"D:\pdf\国赛\国赛历年真题\2023年赛题\C题"
FIG_DIR = os.path.join(BASE, "output", "figures")
TAB_DIR = os.path.join(BASE, "output", "tables")
DATA_DIR = os.path.join(BASE, "solve", "cache")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def setup_chinese_font():
    """注册系统中文字体并设置 rcParams。"""
    for fp in fm.findSystemFonts():
        if any(k in fp for k in ["SimHei", "simhei", "msyh", "YaHei", "SimSun", "simsun"]):
            try:
                fm.fontManager.addfont(fp)
            except Exception:
                pass
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    # 检查是否真的找到了中文字体
    found = [f.name for f in fm.fontManager.ttflist if "SimHei" in f.name or "YaHei" in f.name or "SimSun" in f.name]
    return found


def savefig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved figure:", name)


def load_items():
    """附件1：单品 -> 品类映射"""
    df = pd.read_excel(os.path.join(BASE, "附件1.xlsx"))
    df.columns = ["单品编码", "单品名称", "分类编码", "分类名称"]
    df["单品编码"] = df["单品编码"].astype(str)
    df["分类编码"] = df["分类编码"].astype(str)
    return df


def load_loss():
    """附件4：单品损耗率 + 小分类损耗率"""
    xl = pd.ExcelFile(os.path.join(BASE, "附件4.xlsx"))
    item_loss = xl.parse("Sheet1")
    item_loss.columns = ["单品编码", "单品名称", "损耗率(%)"]
    item_loss["单品编码"] = item_loss["单品编码"].astype(str)
    cat_loss = xl.parse("平均损耗率(%)_小分类编码_不同值")
    cat_loss.columns = ["小分类编码", "小分类名称", "平均损耗率(%)"]
    cat_loss["小分类编码"] = cat_loss["小分类编码"].astype(str)
    return item_loss, cat_loss


def load_wholesale():
    """附件3：批发价格（日期 x 单品）"""
    df = pd.read_excel(os.path.join(BASE, "附件3.xlsx"))
    df.columns = ["日期", "单品编码", "批发价格(元/千克)"]
    df["日期"] = pd.to_datetime(df["日期"])
    df["单品编码"] = df["单品编码"].astype(str)
    return df


def load_sales():
    """附件2：销售流水（全量），返回清洗后的明细。"""
    df = pd.read_excel(os.path.join(BASE, "附件2.xlsx"))
    df.columns = ["销售日期", "扫码销售时间", "单品编码", "销量(千克)", "销售单价(元/千克)", "销售类型", "是否打折销售"]
    df["销售日期"] = pd.to_datetime(df["销售日期"])
    df["单品编码"] = df["单品编码"].astype(str)
    return df
