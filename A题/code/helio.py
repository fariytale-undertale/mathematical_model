# -*- coding: utf-8 -*-
"""
太阳几何位置与 DNI 计算模块
2023 CUMCM A题 定日镜场的优化设计
坐标约定: x 正东, y 正北, z 向上; 方位角从正北顺时针(北0 东90 南180 西270)
"""
import numpy as np
from datetime import date

# ---------- 题目常量 ----------
LAT   = 39.4          # 纬度 (度), 北纬为正
ALTITUDE = 3.0        # 海拔 (km)
G0    = 1.366         # 太阳常数 kW/m2
RHO_REFLECT = 0.92    # 镜面反射率

# 当地时间: 每月21日 9:00 / 10:30 / 12:00 / 13:30 / 15:00
TIMES = [9.0, 10.5, 12.0, 13.5, 15.0]
N_TIMES = len(TIMES)

# 12 个月 21 日的 D 值 (以春分 3月21日 为第0天)
_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
def _days_since_equinox(month):
    """2023 年 (非闰年) 每月 21 日相对春分(3/21) 的天数 D"""
    return (date(2023, month, 21) - date(2023, 3, 21)).days

D_DAYS = np.array([_days_since_equinox(m) for m in _MONTHS], dtype=float)


def solar_declination(D):
    """太阳赤纬角 delta (弧度). 严格按题目 sin(delta)=sin(2piD/365)sin(23.45deg)"""
    D = np.asarray(D, dtype=float)
    return np.arcsin(np.sin(2 * np.pi * D / 365.0) * np.sin(np.deg2rad(23.45)))


def hour_angle(ST):
    """太阳时角 omega (弧度) = pi/12*(ST-12)"""
    return np.pi / 12.0 * (np.asarray(ST, dtype=float) - 12.0)


def solar_position(D, ST):
    """
    返回 (alpha_s, gamma_s) 单位弧度
    alpha_s: 太阳高度角 (0~pi/2, 地平以上)
    gamma_s: 太阳方位角 (0~2pi, 正北顺时针)
    """
    delta = solar_declination(D)
    omega = hour_angle(ST)
    phi = np.deg2rad(LAT)

    sin_alpha = np.cos(delta) * np.cos(phi) * np.cos(omega) + np.sin(delta) * np.sin(phi)
    sin_alpha = np.clip(sin_alpha, -1.0, 1.0)
    alpha_s = np.arcsin(sin_alpha)

    cos_alpha = np.cos(alpha_s)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    # cos(gamma_s) = (sin(delta) - sin(alpha_s)sin(phi)) / (cos(alpha_s)cos(phi))
    cos_gamma = (np.sin(delta) - sin_alpha * sin_phi) / (cos_alpha * cos_phi)
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
    gamma_acos = np.arccos(cos_gamma)          # 0~pi
    # 上午(omega<0)太阳偏东: 方位角<pi; 下午(omega>0)偏西: 方位角>pi
    gamma_s = np.where(omega < 0, gamma_acos, 2 * np.pi - gamma_acos)
    return alpha_s, gamma_s


def sun_direction_vector(D, ST):
    """从镜面指向太阳的单位向量 s (x东 y北 z上)"""
    alpha_s, gamma_s = solar_position(D, ST)
    sx = np.cos(alpha_s) * np.sin(gamma_s)
    sy = np.cos(alpha_s) * np.cos(gamma_s)
    sz = np.sin(alpha_s)
    return np.stack([sx, sy, sz], axis=-1)


def dni(D, ST):
    """法向直接辐射辐照度 DNI (kW/m2)"""
    H = ALTITUDE
    a = 0.4237 - 0.00821 * (6 - H) ** 2
    b = 0.5055 + 0.00595 * (6.5 - H) ** 2
    c = 0.2711 + 0.01858 * (2.5 - H) ** 2
    alpha_s, _ = solar_position(D, ST)
    sin_alpha = np.sin(alpha_s)
    # 日出前/日落后 sin_alpha<=0 -> DNI=0
    dni_val = np.where(sin_alpha > 0, G0 * (a + b * np.exp(-c / np.maximum(sin_alpha, 1e-6))), 0.0)
    return dni_val


def dni_coefficients():
    """返回 a,b,c 三个系数 (海拔固定 3km)"""
    H = ALTITUDE
    a = 0.4237 - 0.00821 * (6 - H) ** 2
    b = 0.5055 + 0.00595 * (6.5 - H) ** 2
    c = 0.2711 + 0.01858 * (2.5 - H) ** 2
    return a, b, c


if __name__ == '__main__':
    print('D 数组 (每月21日):', D_DAYS)
    a, b, c = dni_coefficients()
    print(f'DNI 系数: a={a:.6f} b={b:.6f} c={c:.6f}')
    print('\n每月21日 正午(ST=12) 太阳高度角 & 方位角 & DNI:')
    for m, D in zip(_MONTHS, D_DAYS):
        al, ga = solar_position(D, 12.0)
        d = dni(D, 12.0)
        print(f'  {m:2d}月21日 D={D:4.0f}  高度角={np.rad2deg(al):6.2f}°  方位角={np.rad2deg(ga):6.2f}°  DNI={d:.3f} kW/m2')
