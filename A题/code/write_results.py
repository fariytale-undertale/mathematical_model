# -*- coding: utf-8 -*-
"""将问题2/3最优布局写入 result2.xlsx / result3.xlsx"""
import os
import numpy as np
import openpyxl

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEADER = ['吸收塔x坐标 (m)', '吸收塔y坐标 (m)', '定日镜序号', '定日镜宽度 (m)',
          '定日镜高度 (m)', '定日镜x坐标 (m)', '定日镜y坐标 (m)', '定日镜z坐标 (m)']


def write_result(path, tower_xy, coords, W, H, h):
    """写入结果文件. W,H,h 可为标量或数组"""
    N = coords.shape[0]
    W = np.broadcast_to(np.asarray(W, dtype=float), (N,))
    H = np.broadcast_to(np.asarray(H, dtype=float), (N,))
    h = np.broadcast_to(np.asarray(h, dtype=float), (N,))
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(HEADER)
    for i in range(N):
        ws.append([tower_xy[0], tower_xy[1], i + 1, W[i], H[i],
                   coords[i, 0], coords[i, 1], h[i]])
    wb.save(path)
    print(f'已写入 {path}: N={N}')


if __name__ == '__main__':
    # 问题2
    if os.path.exists(os.path.join(DATA_DIR, 'output', 'problem2_best.npy')):
        x = np.load(os.path.join(DATA_DIR, 'output', 'problem2_best.npy'))
        from problem2 import decode
        d = decode(x)
        write_result(os.path.join(DATA_DIR, 'result2.xlsx'),
                     (d['xt'], d['yt']), d['coords'], d['W'], d['H'], d['h'])
    # 问题3
    if os.path.exists(os.path.join(DATA_DIR, 'output', 'problem3_best.npy')):
        x = np.load(os.path.join(DATA_DIR, 'output', 'problem3_best.npy'))
        from problem3 import decode
        d = decode(x)
        write_result(os.path.join(DATA_DIR, 'result3.xlsx'),
                     (d['xt'], d['yt']), d['coords'], d['W'], d['H'], d['h'])
