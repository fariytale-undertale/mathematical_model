# -*- coding: utf-8 -*-
"""共享数据加载与预处理工具 (问题1-4 复用)"""
import re
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, "output", "figures")
RES_DIR = os.path.join(BASE, "output")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)


def setup_chinese_font():
    for font_path in fm.findSystemFonts():
        if any(name in font_path for name in ['SimHei', 'simhei', 'msyh', 'YaHei', 'SimSun', 'simsun']):
            try:
                fm.fontManager.addfont(font_path)
            except Exception:
                pass
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def week_to_num(s):
    """孕周字符串转连续数值: '11w+6' / '16W+1' -> 11.857 / 16.143"""
    s = str(s).strip().lower()
    m = re.match(r'(\d+)\s*w\s*\+?\s*(\d*)', s)
    if not m:
        return np.nan
    w = int(m.group(1))
    d = int(m.group(2)) if m.group(2) else 0
    return w + d / 7.0


def load_male_data():
    """加载男胎检测数据, 返回预处理后的 DataFrame"""
    wb = pd.ExcelFile(os.path.join(BASE, "附件.xlsx"))
    df = wb.parse("男胎检测数据")
    df = df.rename(columns={
        df.columns[1]: "code", df.columns[2]: "age", df.columns[3]: "height",
        df.columns[4]: "weight", df.columns[9]: "week_str", df.columns[10]: "BMI",
        df.columns[21]: "Y_conc",
    })
    df["week"] = df["week_str"].apply(week_to_num)
    df["lnY"] = np.log(df["Y_conc"])
    df["week_c"] = df["week"] - df["week"].mean()
    df["BMI_c"] = df["BMI"] - df["BMI"].mean()
    return df


def load_female_data():
    """加载女胎检测数据, 返回预处理后的 DataFrame (含问题4特征与标签)"""
    wb = pd.ExcelFile(os.path.join(BASE, "附件.xlsx"))
    df = wb.parse("女胎检测数据")
    df = df.rename(columns={
        df.columns[1]: "code", df.columns[2]: "age", df.columns[3]: "height",
        df.columns[4]: "weight", df.columns[9]: "week_str", df.columns[10]: "BMI",
        df.columns[11]: "reads_total", df.columns[12]: "map_ratio",
        df.columns[13]: "dup_ratio", df.columns[14]: "unique_reads",
        df.columns[15]: "GC", df.columns[16]: "Z13", df.columns[17]: "Z18",
        df.columns[18]: "Z21", df.columns[19]: "ZX", df.columns[22]: "X_conc",
        df.columns[23]: "GC13", df.columns[24]: "GC18", df.columns[25]: "GC21",
        df.columns[26]: "filter_ratio", df.columns[27]: "aneuploidy",
    })
    df["week"] = df["week_str"].apply(week_to_num)
    # 标签: 非整倍体(异常) -> 1, 空白(正常) -> 0
    df["label"] = df["aneuploidy"].notna().astype(int)
    return df
