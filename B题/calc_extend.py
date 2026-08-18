# -*- coding: utf-8 -*-
"""拓展计算: 问题四 DP 精确解, 问题二连续β, 问题一 W(D,α) 可视化"""
import numpy as np, openpyxl, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
for fp in fm.findSystemFonts():
    if any(k in fp.lower() for k in ['simhei','msyh','yahei','simsun']):
        try: fm.fontManager.addfont(fp)
        except Exception: pass
plt.rcParams['font.sans-serif']=['SimHei','Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False

NM=1852.0; theta=120.0
h=np.radians(theta/2); sh,ch=np.sin(h),np.cos(h)

# ============ 1. 问题四动态规划精确解 ============
wb=openpyxl.load_workbook('附件.xlsx',data_only=True); ws=wb['Sheet1']
rows=list(ws.iter_rows(values_only=True))
x=np.array([v for v in rows[1][2:] if v is not None],float)
y=np.array([r[1] for r in rows[2:] if r[1] is not None],float)
Z=np.array([[v for v in r[2:] if v is not None] for r in rows[2:] if r[1] is not None],float)
ny,nx=Z.shape
dzdx=np.gradient(Z,(x[1]-x[0])*NM,axis=1)
tanax=np.abs(dzdx)
wd=Z*sh/(ch-sh*tanax); ws_=Z*sh/(ch+sh*tanax)
ed=dzdx>0
we=np.where(ed,wd,ws_); ww=np.where(ed,ws_,wd)

east = x*NM + we   # east[y,i] 测线i东缘
west = x*NM - ww   # west[y,i] 测线i西缘

K=nx
# no_gap[i,j] = 所有 y 满足 east[:,i] >= west[:,j]
# 预计算: 对每对 (i,j), 检查
print("计算 DP 转移 (O(K^2 * ny))...")
no_gap = np.zeros((K,K), bool)
for i in range(K):
    for j in range(i+1, K):
        no_gap[i,j] = bool(np.all(east[:,i] >= west[:,j] - 1e-6))

INF=10**9
dp=np.full(K, INF, dtype=int)
dp[0]=1
for j in range(1,K):
    cands=[dp[i]+1 for i in range(j) if no_gap[i,j]]
    if cands: dp[j]=min(cands)

# 末条测线覆盖东边界: min_y east >= 4*NM
east_min = east.min(axis=0)
final_candidates = [dp[j] for j in range(K) if east_min[j] >= x[-1]*NM]
dp_N = min(final_candidates)
print(f"问题四 DP(不漏测硬约束) 最少测线数 = {dp_N}")
print(f"策略A 贪心结果 = 82 条, DP 全局最优 = {dp_N} 条")
print(f"贪心与DP一致: {'是' if dp_N==82 else '否, DP更优'}")

# ============ 2. 问题二 连续 β 覆盖宽度 ============
beta_c = np.linspace(0,360,721)
D0=120; alpha=1.5
tan_a=np.tan(np.radians(alpha))
Wb = np.zeros_like(beta_c)
for i,b in enumerate(beta_c):
    tae = tan_a*abs(np.sin(np.radians(b)))
    Wb[i] = D0*sh*(1/(ch-sh*tae)+1/(ch+sh*tae))
fig,ax=plt.subplots(figsize=(7,4))
ax.plot(beta_c, Wb, lw=1.5, color='tab:blue')
ax.axhline(2*D0*np.tan(np.radians(theta/2)), color='gray', ls='--', lw=0.8, label='平坦海底 $2D\\tan(\\theta/2)$')
ax.set_xlabel('测线方向夹角 $\\beta$ / °')
ax.set_ylabel('覆盖宽度 W / m')
ax.set_title('问题二：覆盖宽度随夹角 $\\beta$ 连续变化（$s=0,\\ D_0=120$ m）')
ax.set_xlim(0,360); ax.legend()
ax.set_xticks(np.arange(0,361,45))
fig.tight_layout(); fig.savefig('output/figures/fig_p2_beta.png', dpi=200); plt.close(fig)
print(f"连续β: W(90°)=W(270°)={Wb[180]:.2f} 最大, W(0°)=W(180°)={Wb[0]:.2f} 最小")

# ============ 3. 问题一 W(D, α) 敏感性 ============
D_grid=np.linspace(20,200,100)
a_grid=np.linspace(0,5,100)
WW=np.zeros((len(a_grid),len(D_grid)))
for i,a in enumerate(a_grid):
    ta=np.tan(np.radians(a))
    for j,D in enumerate(D_grid):
        WW[i,j]=D*sh*(1/(ch-sh*ta)+1/(ch+sh*ta))
fig,ax=plt.subplots(figsize=(7,4.5))
im=ax.contourf(D_grid,a_grid,WW,levels=20,cmap='viridis')
cs=ax.contour(D_grid,a_grid,WW,levels=np.arange(100,700,100),colors='white',linewidths=0.6)
ax.clabel(cs,inline=True,fontsize=7,fmt='%dm')
ax.set_xlabel('水深 D / m'); ax.set_ylabel('坡度 $\\alpha$ / °')
ax.set_title('问题一：覆盖宽度 $W(D,\\alpha)$（$\\theta=120^\\circ$）')
fig.colorbar(im,ax=ax,label='覆盖宽度 / m',shrink=0.8)
fig.tight_layout(); fig.savefig('output/figures/fig_p1_WDa.png', dpi=200); plt.close(fig)
print(f"W(D=70,α=0)={WW[0, np.argmin(abs(D_grid-70))]:.2f}, W(D=70,α=1.5)={WW[np.argmin(abs(a_grid-1.5)), np.argmin(abs(D_grid-70))]:.2f}")

print("\n全部拓展计算完成")
