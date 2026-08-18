# -*- coding: utf-8 -*-
"""
问题 4: 基于附件真实海底地形设计多波束测线
三种间距策略对比 + 输出指标
  A: 不漏测(硬)  B: 中位数覆盖(重叠率中位数≈20%)  C: 最浅处(重叠≤20%硬)
"""
import numpy as np
import openpyxl
import json

NM = 1852.0
theta = 120.0
h = np.radians(theta / 2); sh, ch = np.sin(h), np.cos(h)

# ---------- 加载数据 ----------
wb = openpyxl.load_workbook("附件.xlsx", data_only=True)
ws = wb["Sheet1"]
rows = list(ws.iter_rows(values_only=True))
x = np.array([v for v in rows[1][2:] if v is not None], float)   # 西→东
y = np.array([r[1] for r in rows[2:] if r[1] is not None], float) # 南→北
Z = np.array([[v for v in r[2:] if v is not None] for r in rows[2:] if r[1] is not None], float)
ny, nx = Z.shape
dx_nm = x[1] - x[0]; dy_nm = y[1] - y[0]

dzdx = np.gradient(Z, dx_nm * NM, axis=1)
tanax = np.abs(dzdx)
w_deep = Z * sh / (ch - sh * tanax)
w_shallow = Z * sh / (ch + sh * tanax)
east_deep = dzdx > 0
w_east = np.where(east_deep, w_deep, w_shallow)
w_west = np.where(east_deep, w_shallow, w_deep)
Wtot = w_east + w_west

# 测线覆盖边缘 (米)
def east_edge(i): return x[i] * NM + w_east[:, i]
def west_edge(i): return x[i] * NM - w_west[:, i]

def evaluate(line_idx):
    """给定测线索引(西→东), 计算三指标"""
    line_idx = np.array(line_idx)
    N = len(line_idx)
    total_len = N * 5.0
    # 漏测
    covered = np.zeros((ny, nx), bool)
    for i in line_idx:
        we = x[i] * NM - w_west[:, i]
        ee = x[i] * NM + w_east[:, i]
        for iy in range(ny):
            covered[iy, :] |= (x * NM >= we[iy]) & (x * NM <= ee[iy])
    missed_pct = (~covered).sum() / (ny * nx) * 100
    # 重叠率
    etas = []
    for k in range(N - 1):
        i, j = line_idx[k], line_idx[k + 1]
        d = (x[j] - x[i]) * NM
        overlap = w_east[:, i] + w_west[:, j] - d
        avg = (Wtot[:, i] + Wtot[:, j]) / 2.0
        etas.append(overlap / avg)
    etas = np.concatenate(etas)
    over20 = np.sum(etas > 0.20) * dy_nm
    return dict(N=N, total_len=total_len, missed_pct=missed_pct,
                over20_len=over20, eta_mean=etas.mean()*100,
                eta_max=etas.max()*100, eta_min=etas.min()*100,
                over20_frac=np.mean(etas>0.2)*100, line_idx=line_idx)

# ---------- 策略 B: 间距 = 0.8 * 当前测线覆盖宽度中位数 ----------
def strategy_B():
    line = [0]
    while True:
        i = line[-1]
        d = 0.80 * np.median(Wtot[:, i])          # 重叠率中位数≈20%
        x_next = x[i] + d / NM
        if x_next >= x[-1]:
            break
        j = int(np.searchsorted(x, x_next))
        j = min(j, nx - 1)
        if j <= i: j = i + 1
        line.append(j)
    return line

# ---------- 策略 A: 不漏测(硬) ----------
def strategy_A():
    line = [0]
    while True:
        i = line[-1]
        Ei = east_edge(i)
        placed = None
        for j in range(nx - 1, i, -1):
            if np.all(Ei + 1e-6 >= west_edge(j)):
                placed = j; break
        if placed is None or placed == i: break
        line.append(placed)
        if x[placed] * NM + w_east[:, placed].min() >= x[-1] * NM: break
    return line

# ---------- 策略 C: 间距 = 0.8 * 最浅处覆盖宽度 ----------
def strategy_C():
    line = [0]
    while True:
        i = line[-1]
        d = 0.80 * Wtot[:, i].min()
        x_next = x[i] + d / NM
        if x_next >= x[-1]: break
        j = min(int(np.searchsorted(x, x_next)), nx - 1)
        if j <= i: j = i + 1
        line.append(j)
    return line

print("=" * 70)
for name, fn in [("A 不漏测(硬)", strategy_A),
                 ("B 中位数(重叠中位≈20%)", strategy_B),
                 ("C 最浅处(重叠≤20%硬)", strategy_C)]:
    r = evaluate(fn())
    print(f"策略 {name}:")
    print(f"  测线数 N={r['N']}, 总长度={r['total_len']:.1f} NM, "
          f"漏测={r['missed_pct']:.2f}%, 超20%重叠长度={r['over20_len']:.1f} NM")
    print(f"  重叠率: min={r['eta_min']:.1f}%, mean={r['eta_mean']:.1f}%, "
          f"max={r['eta_max']:.1f}%, >20%占比={r['over20_frac']:.1f}%")

# 选定策略 B 作为最终方案
r = evaluate(strategy_B())
print("\n" + "=" * 70)
print("【最终方案: 策略 B】")
print(f"测线数 N = {r['N']}")
print(f"总长度 = {r['total_len']:.1f} NM")
print(f"漏测面积百分比 = {r['missed_pct']:.3f}%")
print(f"重叠率>20% 部分总长度 = {r['over20_len']:.2f} NM")
print("测线位置(海里, 西→东):")
print("  ".join(f"{x[i]:.2f}" for i in r['line_idx']))

json.dump(dict(
    N=r['N'], total_len_NM=r['total_len'], missed_pct=r['missed_pct'],
    over20_len_NM=r['over20_len'], eta_mean=r['eta_mean'], eta_max=r['eta_max'],
    eta_min=r['eta_min'], over20_frac=r['over20_frac'],
    line_x_NM=[float(x[i]) for i in r['line_idx']]),
    open("result4.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("已保存 result4.json")
