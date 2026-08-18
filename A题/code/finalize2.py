# -*- coding: utf-8 -*-
"""问题2 最终结果: 完整验证 + 保存 + 写 result2.xlsx"""
import os
import json
import numpy as np
from layout import generate_layout
from verify import verify_layout
from write_results import write_result

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 最终参数
tower = (-4.99, -0.89)
r0 = 101.73
W, H, h = 6.30, 6.2685, 3.30
ds = 11.35
dr = np.sqrt(3) / 2.0 * ds

coords = generate_layout(tower[0], tower[1], r0, dr, ds)
N = coords.shape[0]
print(f'布局: N={N}, 总面积={N*W*H:.1f} m2')

annual, monthly = verify_layout(coords, W, H, h, tower, K=150)
print(f'完整验证: eta={annual["eta"]:.5f} cos={annual["cos"]:.5f} sb={annual["sb"]:.5f} trunc={annual["trunc"]:.5f}')
print(f'E={annual["E_kw"]/1000:.4f} MW  unit={annual["unit"]:.5f} kW/m2')

# 保存结果
result = dict(
    tower=tower, W=W, H=H, h=h, r0=r0, ds=ds, N=N,
    annual=annual, monthly=monthly,
)
with open(os.path.join(DATA_DIR, 'output', 'problem2_result.json'), 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# 写 result2.xlsx
write_result(os.path.join(DATA_DIR, 'result2.xlsx'), tower, coords, W, H, h)

# 保存参数供后续
np.save(os.path.join(DATA_DIR, 'output', 'problem2_best.npy'),
        np.array([tower[0], tower[1], W, H, h, r0, ds]))
np.save(os.path.join(DATA_DIR, 'output', 'problem2_coords.npy'), coords)
print('问题2 最终结果已保存')
print('逐月:')
for mi in range(12):
    m = monthly[mi]
    print(f'  {mi+1:2d}月 eta={m["eta"]:.5f} cos={m["cos"]:.5f} sb={m["sb"]:.5f} trunc={m["trunc"]:.5f} E={m["E_kw"]/1000:.3f}MW unit={m["unit"]:.5f}')
