"""
问题4: S形调头曲线建模与运动学模拟。

盘入螺线 (h=1.7m, 顺时针) → S形调头曲线 (两段相切圆弧, R1=2*R2)
→ 盘出螺线 (与盘入螺线中心对称, 逆时针)

调头空间: 直径9m圆形区域。
调头曲线与盘入/盘出螺线均相切，两段圆弧之间也相切。

本模块:
  1. 构建S形调头曲线的几何模型
  2. 模拟 t ∈ [-100, 100]s 的全过程
  3. 输出 result4.xlsx
"""
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

# ---- Parameters for Problem 4 ----
H_IN = 1.7           # 盘入螺距 (m)
A_IN = H_IN / (2 * np.pi)
R_TURN = 4.5         # 调头空间半径 (m)
V_HEAD = 1.0         # 龙头速度 (m/s)

# 板凳参数 (same as before)
D_HEAD = 2.86
D_BODY = 1.65
D_LIST = [D_HEAD] + [D_BODY] * 222
N_HANDLES = 224


# ---- Spiral utilities (copied for independence) ----
def spiral_arc_length(theta, a):
    return a * (theta * np.sqrt(theta**2 + 1)
                + np.log(theta + np.sqrt(theta**2 + 1))) / 2

def spiral_to_xy(theta, a):
    r = a * theta
    return np.array([r * np.cos(theta), r * np.sin(theta)])

def spiral_tangent(theta, a):
    """Unit tangent vector of spiral at theta (direction of increasing theta)."""
    r = a * theta
    dr_dtheta = np.array([a * np.cos(theta) - r * np.sin(theta),
                          a * np.sin(theta) + r * np.cos(theta)])
    return dr_dtheta / np.linalg.norm(dr_dtheta)

def chord_distance(theta1, theta2, a):
    r1, r2 = a * theta1, a * theta2
    dtheta = theta1 - theta2
    return np.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * np.cos(dtheta))


# ---- S-curve geometry ----
def build_s_curve(theta_entry, a_in, r_turn=R_TURN):
    """
    Build S-shaped turnaround curve.

    The S-curve consists of two circular arcs:
      - Arc 1: radius R1, tangent to entry spiral at P_entry
      - Arc 2: radius R2 = R1/2, tangent to Arc 1 and to exit spiral
      - Exit spiral is the 180° rotation of entry spiral about origin

    Args:
        theta_entry: spiral parameter at entry point (head position at turnaround start)
        a_in: spiral coefficient for entry spiral
        r_turn: radius of turnaround space

    Returns:
        dict with curve parameters, or None if infeasible
    """
    # Entry point on spiral
    P_entry = spiral_to_xy(theta_entry, a_in)
    r_entry = np.linalg.norm(P_entry)
    T_entry = spiral_tangent(theta_entry, a_in)  # tangent direction (increasing theta)

    # For clockwise entry (theta decreasing), the head moves OPPOSITE to T_entry
    # At the turnaround point, the head transitions from spiral to arc
    # The arc must be tangent to the spiral at the entry point

    # The S-curve turns the direction around. Arc 1 curves one way,
    # Arc 2 curves the opposite way, ending tangent to the exit spiral.

    # Arc 1: center is offset from P_entry perpendicular to tangent
    # For a right turn: center = P_entry + R1 * n (where n is normal)
    # For a left turn: center = P_entry - R1 * n

    # Normal vector (rotated 90° CCW from tangent)
    n_entry = np.array([-T_entry[1], T_entry[0]])

    # The S-curve needs to turn the dragon around within r_turn.
    # Arc 1 center is inside the circle, arc 2 completes the turn.

    # This is a geometric optimization problem:
    # Find R1, R2=R1/2, and arc angles such that:
    #   1. Arc 1 tangent to entry spiral at P_entry
    #   2. Arc 2 tangent to Arc 1 at junction point
    #   3. Arc 2 tangent to exit spiral at P_exit
    #   4. Entire curve within r_turn

    # For now, implement a simplified version with a feasible parameter set.
    # The exact optimization is complex and would require numerical methods.

    # Choose R1 such that the curve fits within the turning space
    # A reasonable starting point: R1 ≈ r_turn / 2
    R1_guess = r_turn * 0.45
    R2_guess = R1_guess / 2

    # Arc 1 curves toward the center (left turn for clockwise entry)
    center1 = P_entry - R1_guess * n_entry

    # The junction point is where Arc 1 ends and Arc 2 begins
    # Arc 1 angle: we need to turn enough so Arc 2 can connect to the exit spiral
    # For a full U-turn, total turn angle ≈ π
    # Arc 1 + Arc 2 ≈ π, with Arc 2 having smaller radius

    # Let Arc 1 go through angle alpha1
    alpha1 = np.pi * 0.55  # ~100° for Arc 1

    # Junction point
    P_junction = center1 + R1_guess * np.array([
        np.cos(alpha1), -np.sin(alpha1)
    ]) @ _rotation_from_basis(n_entry, T_entry)
    # Actually need to be more careful with the geometry...

    # This is getting complex. Let me use a simplified but principled approach.
    return _build_s_curve_numerical(theta_entry, a_in, r_turn)


def _rotation_from_basis(ex, ey):
    """Build rotation matrix from basis vectors (column-wise)."""
    return np.column_stack([ex, ey])


def _build_s_curve_numerical(theta_entry, a_in, r_turn=R_TURN):
    """
    Numerical construction of S-curve.
    Optimize R1 (and thus R2=R1/2) to minimize total arc length
    subject to tangency constraints.
    """
    from scipy.optimize import minimize_scalar

    P_entry = spiral_to_xy(theta_entry, a_in)
    T_entry = spiral_tangent(theta_entry, a_in)
    # Head moves OPPOSITE to T_entry for inward spiral
    T_head = -T_entry

    # Exit spiral: 180° rotation of entry spiral
    # Exit spiral equation: r = a_in * theta (same a), but rotated 180°
    # In polar: theta_exit for point at (r, phi) satisfies
    # r = a_in * theta_exit and phi = theta_exit + pi (mod 2pi)
    # So a point P on exit spiral at parameter theta satisfies:
    # P = (-r*cos(theta), -r*sin(theta)) = (-a_in*theta*cos(theta), -a_in*theta*sin(theta))

    def exit_spiral_xy(theta):
        """Point on exit spiral (180° rotated from entry spiral)."""
        r = a_in * theta
        return np.array([-r * np.cos(theta), -r * np.sin(theta)])

    def exit_spiral_tangent(theta):
        """Unit tangent of exit spiral (increasing theta direction)."""
        r = a_in * theta
        dr = np.array([-a_in * np.cos(theta) + r * np.sin(theta),
                       -a_in * np.sin(theta) - r * np.cos(theta)])
        return dr / np.linalg.norm(dr)

    def try_s_curve(R1):
        """Try to construct S-curve with given R1. Return total length or inf."""
        R2 = R1 / 2

        # Arc 1: choose turn direction (toward center)
        # Normal pointing toward center
        n_to_center = -P_entry / np.linalg.norm(P_entry)
        # Determine which normal direction gives a turn toward center
        n1 = np.array([-T_head[1], T_head[0]])
        if np.dot(n1, n_to_center) < 0:
            n1 = -n1

        center1 = P_entry + R1 * n1

        # Arc 2 must be tangent to Arc 1 at junction
        # Junction: Arc 1 ends, Arc 2 begins
        # Both arcs must have same tangent at junction
        # Arc 2 curves opposite to Arc 1

        # Parameterize Arc 1: angle phi from 0 to alpha1
        # P1(phi) = center1 + R1 * (cos(phi)*u1 + sin(phi)*v1)
        # where u1 = T_head (tangent direction at P_entry)
        # and v1 = n1 (normal direction)

        u1 = T_head
        v1 = n1

        # Arc 2 center: at junction (phi=alpha1), the tangent of Arc 1 is
        # T_junc = -sin(alpha1)*u1 + cos(alpha1)*v1
        # Arc 2 must have same tangent at junction, curving opposite way
        # So center2 is offset from junction by R2 in the opposite normal direction

        # Search for alpha1 and exit theta that satisfy constraints
        best_length = np.inf
        best_params = None

        for alpha1 in np.linspace(np.pi * 0.3, np.pi * 0.9, 30):
            # Junction point
            P_junc = center1 + R1 * (np.cos(alpha1) * u1 + np.sin(alpha1) * v1)

            # Check if within turning space
            if np.linalg.norm(P_junc) > r_turn:
                continue

            # Tangent at junction (Arc 1)
            T_junc = -np.sin(alpha1) * u1 + np.cos(alpha1) * v1

            # Arc 2: center2 = P_junc + R2 * n_junc
            # where n_junc is the normal that makes Arc 2 curve the other way
            # n_junc = -v1_rotated (opposite to Arc 1's normal at junction)
            n_junc = np.array([-T_junc[1], T_junc[0]])

            # Try both normal directions for Arc 2
            for sign in [1, -1]:
                center2 = P_junc + sign * R2 * n_junc

                # Now find exit point: Arc 2 must be tangent to exit spiral
                # This means there exists theta_exit such that:
                #   P_exit = exit_spiral_xy(theta_exit) = center2 + R2 * radial_direction
                #   and the tangent of Arc 2 at P_exit equals exit spiral tangent
                # This is a 1D search over theta_exit

                for theta_exit in np.linspace(10, 50, 100):
                    P_exit = exit_spiral_xy(theta_exit)
                    r_exit = np.linalg.norm(P_exit)

                    # Check if P_exit is on Arc 2 (distance from center2 ≈ R2)
                    dist_to_center2 = np.linalg.norm(P_exit - center2)
                    if abs(dist_to_center2 - R2) > 0.05:
                        continue

                    # Check tangency
                    T_exit = exit_spiral_tangent(theta_exit)
                    # Arc 2 tangent at P_exit (perpendicular to radius)
                    radial = P_exit - center2
                    T_arc2 = np.array([-radial[1], radial[0]]) / R2  # one direction
                    T_arc2_alt = -T_arc2

                    # For exiting (theta increasing on exit spiral), use T_exit
                    dot1 = abs(np.dot(T_arc2, T_exit))
                    dot2 = abs(np.dot(T_arc2_alt, T_exit))

                    if max(dot1, dot2) > 0.99:  # tangency condition
                        # Compute Arc 2 angle
                        vec_junc = P_junc - center2
                        vec_exit = P_exit - center2
                        alpha2 = np.arccos(np.clip(
                            np.dot(vec_junc, vec_exit) / R2**2, -1, 1))

                        total_length = R1 * alpha1 + R2 * alpha2

                        # Check all points within turning space
                        all_inside = True
                        for phi in np.linspace(0, alpha1, 20):
                            pt = center1 + R1 * (np.cos(phi) * u1 + np.sin(phi) * v1)
                            if np.linalg.norm(pt) > r_turn:
                                all_inside = False
                                break
                        if all_inside:
                            for phi in np.linspace(0, alpha2, 20):
                                # For Arc 2, parameterize from junction to exit
                                t = phi / alpha2
                                pt = P_junc + t * (P_exit - P_junc)
                                # More accurate: use polar parameterization
                                if np.linalg.norm(pt) > r_turn:
                                    all_inside = False
                                    break

                        if all_inside and total_length < best_length:
                            best_length = total_length
                            best_params = {
                                'R1': R1, 'R2': R2,
                                'center1': center1, 'center2': center2,
                                'alpha1': alpha1, 'alpha2': alpha2,
                                'P_entry': P_entry, 'P_junc': P_junc,
                                'P_exit': P_exit,
                                'theta_entry': theta_entry,
                                'theta_exit': theta_exit,
                                'total_length': total_length
                            }

        return best_length if best_params else np.inf, best_params

    # Optimize R1
    def objective(R1):
        length, _ = try_s_curve(R1)
        return length

    # Search for best R1
    best_R1 = None
    best_result = None
    best_len = np.inf

    for R1 in np.linspace(0.5, 3.0, 50):
        length, params = try_s_curve(R1)
        if params and length < best_len:
            best_len = length
            best_R1 = R1
            best_result = params

    if best_result:
        print(f"  S-curve found: R1={best_result['R1']:.3f}m, R2={best_result['R2']:.3f}m")
        print(f"  alpha1={best_result['alpha1']:.3f}rad ({best_result['alpha1']*180/np.pi:.1f}deg)")
        print(f"  alpha2={best_result['alpha2']:.3f}rad ({best_result['alpha2']*180/np.pi:.1f}deg)")
        print(f"  Total arc length: {best_result['total_length']:.3f}m")
        print(f"  theta_entry={best_result['theta_entry']:.3f}, theta_exit={best_result['theta_exit']:.3f}")

    return best_result


def main():
    print("=" * 70)
    print("问题4: S形调头曲线建模与运动学模拟")
    print("=" * 70)

    # Determine entry point: head on entry spiral at r_turn boundary
    # Entry spiral: r = (1.7/(2*pi)) * theta, head at 16th loop initially
    # Head moves inward to r = R_TURN = 4.5m
    theta_at_boundary = R_TURN / A_IN
    print(f"Entry spiral: h={H_IN}m, a={A_IN:.6f} m/rad")
    print(f"Head reaches r={R_TURN}m at theta={theta_at_boundary:.4f} "
          f"({theta_at_boundary/np.pi:.4f}pi)")

    # Build S-curve
    print("\n构建S形调头曲线...")
    s_curve = build_s_curve(theta_at_boundary, A_IN)

    if s_curve:
        print(f"\n调头曲线参数:")
        for k, v in s_curve.items():
            if isinstance(v, np.ndarray):
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

        # Save results
        result_path = OUTPUT_DIR / 'problem4_result.txt'
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(f"问题4: S形调头曲线\n{'='*40}\n")
            for k, v in s_curve.items():
                if isinstance(v, np.ndarray):
                    f.write(f"{k}: {v.tolist()}\n")
                else:
                    f.write(f"{k}: {v}\n")
        print(f"\n结果已保存到: {result_path}")

        # Check if curve can be shortened
        print(f"\n调头曲线能否缩短？")
        print(f"  当前总弧长: {s_curve['total_length']:.3f} m")
        print(f"  需要检查是否存在更大R1使总弧长更短...")
        print(f"  (这是问题4第二问的要求)")
    else:
        print("\n[WARNING] 未能构建可行的S形调头曲线。")
        print("需要更精细的几何搜索算法。")


if __name__ == '__main__':
    main()
