# -*- coding: utf-8 -*-
"""生成论文所有图表"""
import numpy as np
import openpyxl
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---------- 中文字体 ----------
for fp in fm.findSystemFonts():
    if any(k in fp.lower() for k in ['simhei', 'msyh', 'yahei', 'simsun']):
        try: fm.fontManager.addfont(fp)
        except Exception: pass
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import os
os.makedirs("output/figures", exist_ok=True)

NM = 1852.0
theta = 120.0
h = np.radians(theta/2); sh, ch = np.sin(h), np.cos(h)

# ================= 加载数据 =================
wb = openpyxl.load_workbook("附件.xlsx", data_only=True); ws = wb["Sheet1"]
rows = list(ws.iter_rows(values_only=True))
x = np.array([v for v in rows[1][2:] if v is not None], float)
y = np.array([r[1] for r in rows[2:] if r[1] is not None], float)
Z = np.array([[v for v in r[2:] if v is not None] for r in rows[2:] if r[1] is not None], float)
ny, nx = Z.shape
dzdx = np.gradient(Z, (x[1]-x[0])*NM, axis=1)

# ---------- 覆盖半宽 ----------
tanax = np.abs(dzdx)
w_deep = Z*sh/(ch - sh*tanax); w_shallow = Z*sh/(ch + sh*tanax)
ed = dzdx > 0
w_east = np.where(ed, w_deep, w_shallow); w_west = np.where(ed, w_shallow, w_deep)
Wtot = w_east + w_west

# ================= 图1: 海底地形 + 测线布设 =================
fig, ax = plt.subplots(figsize=(8, 6))
cm = ax.pcolormesh(x, y, Z, cmap='viridis', shading='auto')
# 等高线
cs = ax.contour(x, y, Z, levels=np.arange(20, 200, 20), colors='white',
                linewidths=0.6, alpha=0.7)
ax.clabel(cs, inline=True, fontsize=7, fmt='%dm')
# 测线
r4 = json.load(open("result4.json", encoding="utf-8"))
for xl in r4['line_x_NM']:
    ax.axvline(xl, color='red', lw=0.6, alpha=0.55)
ax.set_xlabel('东西方向 / 海里'); ax.set_ylabel('南北方向 / 海里')
ax.set_title('附件海域海底地形与多波束测线布设')
ax.set_xlim(0, 4); ax.set_ylim(0, 5); ax.set_aspect('equal')
fig.colorbar(cm, ax=ax, label='海水深度 / m', shrink=0.8)
fig.tight_layout(); fig.savefig('output/figures/fig_terrain.png', dpi=200); plt.close(fig)

# ================= 图2: 问题1 覆盖宽度与重叠率 =================
x1 = np.arange(-800, 801, 200)
D1 = 70 + x1*np.tan(np.radians(1.5))
tan_a = np.tan(np.radians(1.5))
W1 = D1*sh*(1/(ch-sh*tan_a)+1/(ch+sh*tan_a))
eta1 = 1 - 2*200/(W1[1:]+W1[:-1])
fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(x1, W1, 'o-', color='tab:blue', label='覆盖宽度 $W$')
ax1.set_xlabel('测线距中心点距离 / m'); ax1.set_ylabel('覆盖宽度 / m', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2 = ax1.twinx()
ax2.plot(x1[1:], eta1*100, 's--', color='tab:red', label='重叠率 $\\eta$')
ax2.axhline(0, color='gray', lw=0.8, ls=':'); ax2.axhline(10, color='gray', lw=0.8, ls=':')
ax2.axhline(20, color='gray', lw=0.8, ls=':')
ax2.set_ylabel('重叠率 / %', color='tab:red'); ax2.tick_params(axis='y', labelcolor='tab:red')
ax1.set_title('问题1：覆盖宽度与重叠率随测线位置变化（$\\theta=120^\\circ,\\alpha=1.5^\\circ$）')
fig.tight_layout(); fig.savefig('output/figures/fig_p1.png', dpi=200); plt.close(fig)

# ================= 图3: 问题2 覆盖宽度热力图 =================
betas = np.arange(0, 360, 45); ss = np.array([0,0.3,0.6,0.9,1.2,1.5,1.8,2.1])
tan_a2 = np.tan(np.radians(1.5))
W2 = np.zeros((len(betas), len(ss)))
for i,b in enumerate(betas):
    for j,s in enumerate(ss):
        D = 120 - s*NM*np.cos(np.radians(b))*tan_a2
        tae = tan_a2*abs(np.sin(np.radians(b)))
        W2[i,j] = D*sh*(1/(ch-sh*tae)+1/(ch+sh*tae))
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(W2, aspect='auto', cmap='viridis', origin='lower')
ax.set_xticks(range(len(ss))); ax.set_xticklabels([f'{s:.1f}' for s in ss])
ax.set_yticks(range(len(betas))); ax.set_yticklabels([f'{b}' for b in betas])
ax.set_xlabel('测量船距中心点距离 / 海里'); ax.set_ylabel('测线方向夹角 $\\beta$ / °')
ax.set_title('问题2：覆盖宽度 $W(\\beta, s)$ / m')
for i in range(len(betas)):
    for j in range(len(ss)):
        ax.text(j, i, f'{W2[i,j]:.0f}', ha='center', va='center', fontsize=7,
                color='white' if W2[i,j] < 500 else 'black')
fig.colorbar(im, ax=ax, label='覆盖宽度 / m', shrink=0.8)
fig.tight_layout(); fig.savefig('output/figures/fig_p2.png', dpi=200); plt.close(fig)

# ================= 图4: 问题3 测线布设(东西剖面) =================
# 均匀斜面, 西深东浅
xw = np.linspace(-3704, 3704, 500)
Dw = 110 - xw*np.tan(np.radians(1.5))   # 水深
seabed = -Dw
# 测线位置(问题3结果)
r3_xs = np.load("solve3_xs.npy") if os.path.exists("solve3_xs.npy") else None
if r3_xs is None:
    # 复现问题3
    from scipy.optimize import brentq
    kd = sh/(ch-sh*tan_a); Lw, Le = -2*NM, 2*NM
    x1p = (Lw + 110*kd)/(1 + tan_a*kd)
    xs3 = [x1p]
    def D3(xx): return 110 - xx*tan_a
    def W3(xx): return D3(xx)*sh*(1/(ch-sh*tan_a)+1/(ch+sh*tan_a))
    while xs3[-1] + D3(xs3[-1])*sh/(ch+sh*tan_a) < Le:
        xi = xs3[-1]
        def f(d):
            xj = xi+d; return (1 - d/((W3(xi)+W3(xj))/2)) - 0.10
        d = brentq(f, 1e-6, 0.95*W3(xi)); xs3.append(xi+d)
    r3_xs = np.array(xs3)
    np.save("solve3_xs.npy", r3_xs)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(xw, seabed, color='saddlebrown', lw=2, label='海底坡面')
for i, xc in enumerate(r3_xs):
    c = 'tab:blue' if i % 2 == 0 else 'tab:cyan'
    wsh = D3(xc)*sh/(ch+sh*tan_a); wdp = D3(xc)*sh/(ch-sh*tan_a)
    ax.plot([xc-wdp, xc+wsh], [-D3(xc), -D3(xc)], color=c, lw=3, alpha=0.7)
ax.plot([], [], color='tab:blue', lw=3, label='覆盖条带')
ax.set_xlabel('东西方向 / m'); ax.set_ylabel('深度 / m')
ax.set_title(f'问题3：均匀斜面测线布设（{len(r3_xs)} 条测线，总长 {len(r3_xs)*2:.0f} 海里）')
ax.legend(loc='lower right'); ax.invert_yaxis()
fig.tight_layout(); fig.savefig('output/figures/fig_p3.png', dpi=200); plt.close(fig)

# ================= 图5: 问题4 三策略 Pareto / k 扫描 =================
ks = np.array([0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95])
Ns = np.array([66,56,52,51,49,47,46,43])
Ls = Ns*5.0
miss = np.array([1.62,2.05,2.93,2.66,4.21,5.54,6.09,8.84])
o20 = np.array([238.9,171.7,133.7,111.8,94.8,80.0,70.7,57.9])
fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(ks, Ls, 'o-', color='tab:blue', label='总长度 (左轴)')
ax1.set_xlabel('间距系数 $k$（间距 = $k\\cdot$覆盖宽度中位数）')
ax1.set_ylabel('测线总长度 / 海里', color='tab:blue'); ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2 = ax1.twinx()
ax2.plot(ks, miss, 's--', color='tab:red', label='漏测率 (右轴)')
ax2.plot(ks, o20/5, '^--', color='tab:green', label='超20%重叠长度/5 (右轴)')
ax2.set_ylabel('漏测率 / %；超20%重叠长度 / 5 海里', color='tab:red'); ax2.tick_params(axis='y', labelcolor='tab:red')
ax1.axvline(0.8, color='gray', ls=':', lw=1)
ax1.text(0.8, 300, ' 选定 k=0.8', fontsize=9, color='gray')
ax1.set_title('问题4：间距系数对三个指标的影响（$k$ 扫描）')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc='center left')
fig.tight_layout(); fig.savefig('output/figures/fig_p4.png', dpi=200); plt.close(fig)

print("图表生成完毕:")
for f in sorted(os.listdir('output/figures')):
    print("  output/figures/"+f)
