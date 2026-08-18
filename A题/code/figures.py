# -*- coding: utf-8 -*-
"""生成论文图表"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---- 中文字体配置 ----
for fp in fm.findSystemFonts():
    if any(n in fp for n in ['SimHei', 'simhei', 'msyh', 'YaHei', 'SimSun', 'simsun']):
        try:
            fm.fontManager.addfont(fp)
        except Exception:
            pass
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(DATA_DIR, 'output', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

MONTHS = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']


def fig_layout(pts, tower_xy=(0, 0), title='问题1 定日镜场布局', fname='fig1_layout.png'):
    """镜场布局图"""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pts[:, 0], pts[:, 1], s=2, c='steelblue', label=f'定日镜 (N={len(pts)})')
    ax.scatter([tower_xy[0]], [tower_xy[1]], marker='*', s=300, c='red', zorder=5, label='吸收塔')
    # 内圈/外圈
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(100*np.cos(th), 100*np.sin(th), 'k--', lw=1, alpha=0.6)
    ax.plot(350*np.cos(th), 350*np.sin(th), 'k-', lw=1.2)
    ax.set_aspect('equal')
    ax.set_xlabel('x (东) / m')
    ax.set_ylabel('y (北) / m')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=300)
    plt.close()


def fig_solar(fname='fig2_solar.png'):
    """全年太阳高度角/方位角与 DNI"""
    from helio import D_DAYS, TIMES, solar_position, dni
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    alpha = np.zeros((12, len(TIMES)))
    gamma = np.zeros((12, len(TIMES)))
    d = np.zeros((12, len(TIMES)))
    for i in range(12):
        for j, st in enumerate(TIMES):
            al, ga = solar_position(D_DAYS[i], st)
            alpha[i, j] = np.rad2deg(al)
            gamma[i, j] = np.rad2deg(ga)
            d[i, j] = dni(D_DAYS[i], st)
    for ax, data, title, cmap in [
        (axes[0], alpha, '太阳高度角 / °', 'viridis'),
        (axes[1], gamma, '太阳方位角 / °', 'plasma'),
        (axes[2], d, 'DNI / (kW/m^2)', 'YlOrRd')]:
        im = ax.imshow(data.T, aspect='auto', cmap=cmap, origin='lower')
        ax.set_xticks(range(12)); ax.set_xticklabels(MONTHS, fontsize=8, rotation=45)
        ax.set_yticks(range(len(TIMES))); ax.set_yticklabels([f'{s:.0f}' for s in TIMES], fontsize=8)
        ax.set_xlabel('月份'); ax.set_ylabel('当地时间')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=300)
    plt.close()


def fig_monthly_eff(res_path, fname='fig3_monthly.png'):
    """问题1 逐月效率分量"""
    with open(res_path, encoding='utf-8') as f:
        res = json.load(f)
    monthly = res['monthly']
    keys = ['cos', 'sb', 'trunc', 'eta']
    labels = ['余弦效率', '阴影遮挡效率', '截断效率', '总光学效率']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, lab, c in zip(keys, labels, colors):
        vals = [monthly[str(i)][k] for i in range(12)]
        ax.plot(range(12), vals, '-o', ms=4, label=lab, color=c)
    ax.set_xticks(range(12)); ax.set_xticklabels(MONTHS)
    ax.set_ylabel('效率')
    ax.set_title('问题1 逐月平均效率分量')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=300)
    plt.close()


if __name__ == '__main__':
    from mirrors import load_mirrors, ATTACH_PATH
    pts = load_mirrors(ATTACH_PATH)
    fig_layout(pts)
    fig_solar()
    fig_monthly_eff(os.path.join(DATA_DIR, 'output', 'problem1_results.json'))
    print('图表已生成:', os.listdir(FIG_DIR))
