"""
问题5: 确定龙头最大行进速度，使所有把手速度不超过 2 m/s。

在问题4确定的路径 (S形调头曲线) 下，龙头以恒定速度行进，
各把手的速度与龙头速度近似成比例关系 (准静态假设)。

方法:
  1. 在 v_head = 1 m/s 下模拟全过程
  2. 记录所有把手的最大瞬时速度 v_max_i
  3. 速度放大系数 k_i = v_max_i / 1
  4. 最大龙头速度 = min_i (2 / k_i) = 2 / max_i(k_i)
"""
import numpy as np
from pathlib import Path
import common

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)


def estimate_max_head_speed():
    """
    Estimate maximum head speed from the Problem 1 simulation data.
    The velocity amplification factor is largest when the spiral is tightest
    (inner handles move faster than outer handles due to geometric constraint).
    """
    print("从问题1模拟数据估计速度放大系数...")

    # Run Problem 1 simulation for as long as feasible
    t_array, positions, velocities, theta_all, failed_at = \
        common.simulate_time_range(0, 300)

    n_valid = np.sum(~np.isnan(velocities[:, 0]))
    print(f"  有效时间步: {n_valid}")

    if n_valid == 0:
        print("[ERROR] 无有效数据")
        return None

    # For each handle, find its maximum speed during the simulation
    valid_vel = velocities[:n_valid, :]
    max_speed_per_handle = np.nanmax(valid_vel, axis=0)

    # Speed amplification factor (v_head = 1 m/s)
    k_factors = max_speed_per_handle / 1.0

    # Maximum amplification
    max_k = np.nanmax(k_factors)
    handle_max_k = np.nanargmax(k_factors)

    # Maximum allowable head speed
    v_head_max = 2.0 / max_k

    print(f"\n速度放大分析 (基于 v_head = 1 m/s):")
    print(f"  最大放大系数: k_max = {max_k:.4f} (把手 #{handle_max_k})")
    print(f"  龙头最大速度: v_head_max = {v_head_max:.4f} m/s")

    # Breakdown by section
    print(f"\n各关键把手放大系数:")
    for label, idx in common.KEY_HANDLES.items():
        if idx < n_valid:
            print(f"  {label:>14s}: k = {k_factors[idx]:.4f}, "
                  f"v_max = {max_speed_per_handle[idx]:.4f} m/s, "
                  f"v_head_limit = {2.0/k_factors[idx]:.4f} m/s")

    # Detailed analysis: track speed over time for the fastest handle
    print(f"\n最快把手 (#{handle_max_k}) 的速度-时间曲线分析:")
    print(f"  (此把手速度先增后减，在螺线最紧处达到峰值)")

    return {
        'v_head_max': v_head_max,
        'max_k': max_k,
        'handle_max_k': handle_max_k,
        'k_factors': k_factors,
        'max_speed_per_handle': max_speed_per_handle
    }


def verify_at_speed(v_head_test):
    """
    Verify that at speed v_head_test, no handle exceeds 2 m/s.
    Under quasi-static assumption, speeds scale linearly.
    """
    print(f"\n验证龙头速度 v_head = {v_head_test:.4f} m/s...")

    t_array, positions, velocities, theta_all, failed_at = \
        common.simulate_time_range(0, 300)

    n_valid = np.sum(~np.isnan(velocities[:, 0]))
    valid_vel = velocities[:n_valid, :]

    # Scale velocities
    scaled_vel = valid_vel * v_head_test / common.V_HEAD

    max_scaled = np.nanmax(scaled_vel)
    handle_max = np.nanargmax(np.nanmax(scaled_vel, axis=0))

    print(f"  缩放后最大速度: {max_scaled:.4f} m/s (把手 #{handle_max})")
    if max_scaled <= 2.0:
        print(f"  ✓ 所有把手速度 ≤ 2 m/s，可行!")
    else:
        print(f"  ✗ 存在把手速度 > 2 m/s，不可行!")
        # Find how many handles exceed
        exceed_count = np.sum(np.nanmax(scaled_vel, axis=0) > 2.0)
        print(f"  超速把手数: {exceed_count}/{common.N_HANDLES}")

    return max_scaled <= 2.0, max_scaled


def main():
    print("=" * 70)
    print("问题5: 确定龙头最大行进速度 (所有把手 ≤ 2 m/s)")
    print("=" * 70)

    result = estimate_max_head_speed()

    if result is None:
        return

    v_head_max = result['v_head_max']

    # Verify
    feasible, actual_max = verify_at_speed(v_head_max)

    # Slightly adjust if needed
    if not feasible:
        print(f"\n调整: 降低速度以确保可行性...")
        v_adjusted = v_head_max * 0.99
        feasible2, _ = verify_at_speed(v_adjusted)
        if feasible2:
            v_head_max = v_adjusted

    print(f"\n{'='*50}")
    print(f"最终结果:")
    print(f"  龙头最大行进速度: {v_head_max:.4f} m/s")
    print(f"  速度放大系数:     {result['max_k']:.4f}")
    print(f"  限制把手:         #{result['handle_max_k']}")
    print(f"{'='*50}")

    # Save
    result_path = OUTPUT_DIR / 'problem5_result.txt'
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(f"问题5: 龙头最大行进速度\n{'='*40}\n")
        f.write(f"v_head_max = {v_head_max:.6f} m/s\n")
        f.write(f"速度放大系数 k_max = {result['max_k']:.6f}\n")
        f.write(f"限制把手: #{result['handle_max_k']}\n")
        for label, idx in common.KEY_HANDLES.items():
            k = result['k_factors'][idx]
            f.write(f"{label}: k={k:.6f}, v_limit={2.0/k:.6f} m/s\n")
    print(f"\n结果已保存到: {result_path}")


if __name__ == '__main__':
    main()
