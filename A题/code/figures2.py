# -*- coding: utf-8 -*-
"""生成问题2/3 图表 (含塔位敏感性)"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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
MONTHS = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']


def fig_p2_layout(fname='fig4_p2_layout.png'):
    coords = np.load(os.path.join(DATA_DIR, 'output', 'problem2_coords.npy'))
    with open(os.path.join(DATA_DIR, 'output', 'problem2_result.json'), encoding='utf-8') as f:
        res = json.load(f)
    tower = res['tower']
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(coords[:, 0], coords[:, 1], s=1.5, c='steelblue', label=f'定日镜 (N={len(coords)})')
    ax.scatter([tower[0]], [tower[1]], marker='*', s=300, c='red', zorder=5, label='吸收塔')
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(100 * np.cos(th) + tower[0], 100 * np.sin(th) + tower[1], 'k--', lw=1, alpha=0.6)
    ax.plot(350 * np.cos(th), 350 * np.sin(th), 'k-', lw=1.2)
    ax.set_aspect('equal')
    ax.set_xlabel('x (东) / m'); ax.set_ylabel('y (北) / m')
    ax.set_title('问题二 最优定日镜场布局')
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=300)
    plt.close()


def fig_p2_monthly(fname='fig5_p2_monthly.png'):
    with open(os.path.join(DATA_DIR, 'output', 'problem2_result.json'), encoding='utf-8') as f:
        res = json.load(f)
    monthly = res['monthly']
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, lab, c in [('cos', '余弦效率', 'tab:blue'), ('sb', '阴影遮挡效率', 'tab:orange'),
                      ('trunc', '截断效率', 'tab:green'), ('eta', '总光学效率', 'tab:red')]:
        vals = [m[k] for m in monthly]
        ax.plot(range(12), vals, '-o', ms=4, label=lab, color=c)
    ax.set_xticks(range(12)); ax.set_xticklabels(MONTHS)
    ax.set_ylabel('效率')
    ax.set_title('问题二 逐月平均效率分量')
    ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=300)
    plt.close()


def fig_compare(fname='fig6_compare.png'):
    with open(os.path.join(DATA_DIR, 'output', 'problem1_results.json'), encoding='utf-8') as f:
        p1 = json.load(f)['annual']
    with open(os.path.join(DATA_DIR, 'output', 'problem2_result.json'), encoding='utf-8') as f:
        p2 = json.load(f)['annual']
    with open(os.path.join(DATA_DIR, 'output', 'problem3_result.json'), encoding='utf-8') as f:
        p3 = json.load(f)['annual']
    labels = ['问题一', '问题二', '问题三']
    E = [p1['E_kw'] / 1000, p2['E_kw'] / 1000, p3['E_kw'] / 1000]
    unit = [p1['unit'], p2['unit'], p3['unit']]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    ax1.bar(labels, E, color=['tab:blue', 'tab:orange', 'tab:green'])
    ax1.axhline(60, color='r', ls='--', lw=1, label='额定 60 MW')
    ax1.set_ylabel('年平均输出热功率 / MW'); ax1.legend(fontsize=8)
    ax2.bar(labels, unit, color=['tab:blue', 'tab:orange', 'tab:green'])
    ax2.set_ylabel('单位面积年平均输出热功率 / (kW/m^2)')
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=300)
    plt.close()


def fig_tower_sensitivity(fname='fig7_tower.png'):
    with open(os.path.join(DATA_DIR, 'output', 'tower_sensitivity.json'), encoding='utf-8') as f:
        res = json.load(f)
    yt = [r['yt'] for r in res]
    cos = [r['cos'] for r in res]
    unit = [r['unit'] for r in res]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(yt, cos, '-o', color='tab:blue', label='余弦效率')
    ax1.set_xlabel('吸收塔纵坐标 y / m'); ax1.set_ylabel('余弦效率', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(yt, unit, '-s', color='tab:red', label='单位面积功率')
    ax2.set_ylabel('单位面积功率 / (kW/m^2)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_title('吸收塔纵向位置对余弦效率与单位面积功率的影响')
    ax1.grid(alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=300)
    plt.close()


if __name__ == '__main__':
    fig_p2_layout()
    fig_p2_monthly()
    fig_compare()
    fig_tower_sensitivity()
    print('图表已生成:', os.listdir(FIG_DIR))
