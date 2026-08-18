"""
2024 CUMCM Problem B: Plotting Utilities
可视化工具 — 中文字体支持
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

# ============================================================
# Robust Chinese Font Setup
# ============================================================

def _setup_chinese_font():
    """Configure matplotlib for Chinese text rendering."""
    # Explicitly register known CJK fonts so matplotlib finds them reliably
    cjk_font_paths = [
        'C:/Windows/Fonts/simhei.ttf',        # 黑体 (preferred)
        'C:/Windows/Fonts/msyh.ttc',          # 微软雅黑
        'C:/Windows/Fonts/simsun.ttc',        # 宋体
    ]

    registered = []
    for fp in cjk_font_paths:
        if os.path.exists(fp):
            try:
                fm.fontManager.addfont(fp)
                registered.append(fp)
            except Exception:
                pass

    # Set the sans-serif font list: CJK font first, then DejaVu for math symbols
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # Rebuild font cache
    try:
        fm._load_fontmanager(try_read_cache=False)
    except Exception:
        pass

    return registered

_registered = _setup_chinese_font()

# ============================================================
# Global Settings
# ============================================================

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette — accessible, distinct
COLORS = ['#2166AC', '#D6604D', '#4DAF4A', '#984EA3', '#FF7F00', '#A65628', '#F781BF', '#999999']


def savefig(fig, name):
    """Save figure to output directory."""
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  [Figure saved] {path}")
    return path


def new_figure(figsize=(10, 6), title=None):
    """Create a new figure with consistent styling."""
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax


def plot_decision_table(ax, decisions_df, title="Decision Scheme"):
    """Plot a decision table as a styled text table on axes."""
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

    col_labels = decisions_df.columns.tolist()
    cell_text = decisions_df.values.tolist()

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.5)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#2166AC')
        table[0, j].set_text_props(color='white', fontweight='bold')

    return table
