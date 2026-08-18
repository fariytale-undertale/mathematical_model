"""
2024 CUMCM Problem A: "板凳龙" 闹元宵 — 公用模块 v2
基于A242优秀论文的方法改进:
  - 碰撞检测: 板凳矩形顶点到相邻圈板凳直线的距离
  - 更鲁棒的把手递推求解
"""
import numpy as np
from pathlib import Path

# ============================================================
# Physical parameters (all in SI: m, s, rad)
# ============================================================
H_PITCH = 0.55                     # 螺距 (m)
A_COEF = H_PITCH / (2 * np.pi)     # 螺线系数 r = a * theta
V_HEAD = 1.0                       # 龙头速度 (m/s)
THETA0 = 32 * np.pi                # 初始龙头角度 (第16圈, 正x轴)
D_HEAD = 3.41 - 2 * 0.275          # 龙头两把手间距 = 2.86 m
D_BODY = 2.20 - 2 * 0.275          # 龙身/龙尾两把手间距 = 1.65 m
BOARD_WIDTH = 0.30                 # 板宽 (m)
HALF_WIDTH = BOARD_WIDTH / 2       # 半宽 = 0.15 m (碰撞安全距离)
HOLE_OFFSET = 0.275                # 孔中心距板头距离 (m)
N_SECTIONS = 223                   # 总板凳数
N_HANDLES = N_SECTIONS + 1         # 总把手数 = 224
D_LIST = [D_HEAD] + [D_BODY] * 222 # 各段板凳把手间距

# ============================================================
# Spiral geometry
# ============================================================
def spiral_arc_length(theta, a=None):
    """Arc length along Archimedean spiral from theta=0 to theta."""
    if a is None:
        a = A_COEF
    return a * (theta * np.sqrt(theta**2 + 1)
                + np.log(theta + np.sqrt(theta**2 + 1))) / 2

def d_arc_dtheta(theta, a=None):
    """ds/dtheta."""
    if a is None:
        a = A_COEF
    return a * np.sqrt(theta**2 + 1)

def spiral_to_xy(theta, a=None):
    """Convert spiral parameter theta to Cartesian (x, y)."""
    if a is None:
        a = A_COEF
    r = a * theta
    return r * np.cos(theta), r * np.sin(theta)

def chord_distance(theta1, theta2, a=None):
    """Chord distance between two points on the spiral."""
    if a is None:
        a = A_COEF
    r1, r2 = a * theta1, a * theta2
    dtheta = theta1 - theta2
    return np.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * np.cos(dtheta))

# ============================================================
# Arc-length inversion
# ============================================================
_S0_INITIAL = spiral_arc_length(THETA0)

def _inv_arc_length(target_s, a=None, theta_low=0.0001, theta_high=None):
    """Find theta such that spiral_arc_length(theta) = target_s."""
    if a is None:
        a = A_COEF
    if target_s <= 0:
        return 0.0001
    if theta_high is None:
        theta_high = THETA0 * 2
    while spiral_arc_length(theta_high, a) < target_s:
        theta_high *= 2
        if theta_high > 1e5:
            break
    lo, hi = theta_low, theta_high
    for _ in range(80):
        mid = (lo + hi) / 2
        if spiral_arc_length(mid, a) < target_s:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14:
            return (lo + hi) / 2
    th = (lo + hi) / 2
    for _ in range(30):
        f_val = spiral_arc_length(th, a) - target_s
        if abs(f_val) < 1e-12:
            break
        th = th - f_val / d_arc_dtheta(th, a)
        if th <= 0:
            th = 0.001
    return th

def head_theta_at_time(t, a=None):
    """Head front-handle theta at time t."""
    if a is None:
        a = A_COEF
    s0 = spiral_arc_length(THETA0, a)
    return _inv_arc_length(s0 - V_HEAD * t, a, theta_low=0.001, theta_high=THETA0)

# ============================================================
# Next-handle finder (improved: wider search for tight spirals)
# ============================================================
def find_next_handle(theta_prev, target_d, a=None, direction='inward'):
    """
    Find theta on spiral such that chord_distance(theta_prev, theta) = target_d.

    CRITICAL INSIGHT from A242 paper: handles are NOT always at monotonically
    decreasing theta. When the dragon coils deeply, some handles wrap around
    the center and emerge at LARGER theta values than previous handles.

    Search strategy: scan wide theta range around theta_prev, find the solution
    that gives the correct chord distance.

    direction: 'inward' (toward smaller theta, default) or 'outward' (larger theta)
    """
    if a is None:
        a = A_COEF
    if theta_prev <= 0.001:
        return None

    # Search range: wide enough to find handles on different loops
    # The chord distance to a point on the spiral oscillates with period 2π
    # Search theta_prev ± 4π to cover multiple nearby loops
    theta_low = max(0.0001, theta_prev - 4 * np.pi)
    theta_high = theta_prev + 4 * np.pi

    # Scan for solutions
    solutions = []
    for n_scan in [3000, 10000]:
        theta_scan = np.linspace(theta_low, theta_high, n_scan)
        for k in range(n_scan - 1):
            dk = chord_distance(theta_prev, theta_scan[k], a)
            dk1 = chord_distance(theta_prev, theta_scan[k + 1], a)
            if (dk - target_d) * (dk1 - target_d) <= 0:
                lo_b, hi_b = theta_scan[k], theta_scan[k + 1]
                for _ in range(50):
                    mid = (lo_b + hi_b) / 2
                    dm = chord_distance(theta_prev, mid, a)
                    if abs(dm - target_d) < 1e-10:
                        solutions.append(mid)
                        break
                    if (dk - target_d) * (dm - target_d) <= 0:
                        hi_b = mid
                    else:
                        lo_b = mid
                    if hi_b - lo_b < 1e-13:
                        solutions.append((lo_b + hi_b) / 2)
                        break
        if solutions:
            break  # Found at least one solution

    if not solutions:
        return None

    # Choose the solution based on direction preference
    if direction == 'inward':
        # Prefer theta < theta_prev (inward = smaller theta)
        inward_sols = [s for s in solutions if s < theta_prev]
        if inward_sols:
            # Choose the closest one (largest theta among inward solutions)
            return max(inward_sols)
        # Fall back to the closest solution overall
        return min(solutions, key=lambda s: abs(s - theta_prev))
    elif direction == 'outward':
        outward_sols = [s for s in solutions if s > theta_prev]
        if outward_sols:
            return min(outward_sols)
        return min(solutions, key=lambda s: abs(s - theta_prev))
    else:
        # Closest solution
        return min(solutions, key=lambda s: abs(s - theta_prev))

# ============================================================
# Full dragon computation
# ============================================================
def _find_next_handle_fast(theta_prev, target_d, a, search_dir):
    """
    Fast Newton-based search for next handle in a known direction.

    search_dir: 'inward' (theta < theta_prev) or 'outward' (theta > theta_prev)
    Returns theta or None.
    """
    if theta_prev <= 0.001:
        return None

    # Arc-length-based initial guess
    s_prev = spiral_arc_length(theta_prev, a)
    if search_dir == 'inward':
        s_target = s_prev - target_d * 1.02
        theta_guess = _inv_arc_length(s_target, a, theta_high=theta_prev) if s_target > 0 else 0.001
        if theta_guess >= theta_prev:
            theta_guess = theta_prev - target_d / d_arc_dtheta(theta_prev, a)
    else:  # outward
        s_target = s_prev + target_d * 1.02
        theta_guess = _inv_arc_length(s_target, a, theta_low=theta_prev, theta_high=theta_prev * 5)
        if theta_guess <= theta_prev:
            theta_guess = theta_prev + target_d / d_arc_dtheta(theta_prev, a)

    # Newton refinement
    if theta_guess <= 0:
        theta_guess = 0.001
    theta = theta_guess
    for _ in range(50):
        d = chord_distance(theta_prev, theta, a)
        f_val = d - target_d
        if abs(f_val) < 1e-10:
            return theta
        eps = max(1e-8, theta * 1e-8)
        dd = (chord_distance(theta_prev, theta + eps, a) - d) / eps
        if abs(dd) < 1e-15:
            break
        step = f_val / dd
        theta_new = theta - step
        # Enforce search direction
        if search_dir == 'inward' and theta_new >= theta_prev:
            theta_new = (theta + theta_prev * 0.1) / 2
        if search_dir == 'outward' and theta_new <= theta_prev:
            theta_new = theta_prev + abs(step) * 0.5
        if theta_new <= 0.0001:
            theta_new = 0.001
        if abs(theta_new - theta) < 1e-13:
            theta = theta_new
            break
        theta = theta_new

    # Narrow scan fallback
    if search_dir == 'inward':
        theta_scan = np.linspace(0.0001, theta_prev, 2000)
    else:
        theta_scan = np.linspace(theta_prev, theta_prev + 2*np.pi, 2000)

    for k in range(len(theta_scan) - 1):
        dk = chord_distance(theta_prev, theta_scan[k], a)
        dk1 = chord_distance(theta_prev, theta_scan[k + 1], a)
        if (dk - target_d) * (dk1 - target_d) <= 0:
            lo_b, hi_b = theta_scan[k], theta_scan[k + 1]
            for _ in range(50):
                mid = (lo_b + hi_b) / 2
                dm = chord_distance(theta_prev, mid, a)
                if abs(dm - target_d) < 1e-10:
                    return mid
                if (dk - target_d) * (dm - target_d) <= 0:
                    hi_b = mid
                else:
                    lo_b = mid
                if hi_b - lo_b < 1e-13:
                    return (lo_b + hi_b) / 2
            return (lo_b + hi_b) / 2
    return None


def compute_dragon_at_time(t, a=None):
    """
    Fast computation of all 224 handle thetas at time t.

    Strategy:
      1. Determine chain direction from head position
      2. Use fast directional Newton search
      3. Fall back to scan search only if Newton fails
    """
    if a is None:
        a = A_COEF
    theta_array = np.full(N_HANDLES, np.nan)
    theta_array[0] = head_theta_at_time(t, a)

    s_head = spiral_arc_length(theta_array[0], a)
    chain_arc_est = sum(D_LIST) * 1.02

    # Primary direction
    primary_dir = 'inward' if s_head > chain_arc_est else 'outward'

    for i in range(1, N_HANDLES):
        th = _find_next_handle_fast(theta_array[i - 1], D_LIST[i - 1], a, primary_dir)
        if th is not None:
            theta_array[i] = th
            continue

        # If primary direction failed, try the other direction
        alt_dir = 'outward' if primary_dir == 'inward' else 'inward'
        th = _find_next_handle_fast(theta_array[i - 1], D_LIST[i - 1], a, alt_dir)
        if th is not None:
            theta_array[i] = th
            primary_dir = alt_dir  # switch for subsequent handles
            continue

        # Both failed — use wide scan as last resort
        sols = find_all_solutions(theta_array[i - 1], D_LIST[i - 1], a)
        if not sols:
            return None, i
        theta_array[i] = min(sols, key=lambda s: abs(s - theta_array[i - 1]))

    return theta_array, None


def find_all_solutions(theta_prev, target_d, a=None):
    """Find ALL theta values on spiral at chord distance target_d from theta_prev."""
    if a is None:
        a = A_COEF
    if theta_prev <= 0.001:
        return []

    # Wide search range covering several loops in both directions
    theta_low = max(0.0001, theta_prev - 6 * np.pi)
    theta_high = theta_prev + 6 * np.pi

    solutions = []
    for n_scan in [3000, 10000]:
        solutions = []
        theta_scan = np.linspace(theta_low, theta_high, n_scan)
        for k in range(n_scan - 1):
            dk = chord_distance(theta_prev, theta_scan[k], a)
            dk1 = chord_distance(theta_prev, theta_scan[k + 1], a)
            if (dk - target_d) * (dk1 - target_d) <= 0:
                lo_b, hi_b = theta_scan[k], theta_scan[k + 1]
                for _ in range(50):
                    mid = (lo_b + hi_b) / 2
                    dm = chord_distance(theta_prev, mid, a)
                    if abs(dm - target_d) < 1e-10:
                        # Check if this is a new solution (not too close to existing)
                        if not solutions or min(abs(mid - s) for s in solutions) > 1e-4:
                            solutions.append(mid)
                        break
                    if (dk - target_d) * (dm - target_d) <= 0:
                        hi_b = mid
                    else:
                        lo_b = mid
                    if hi_b - lo_b < 1e-13:
                        mid_val = (lo_b + hi_b) / 2
                        if not solutions or min(abs(mid_val - s) for s in solutions) > 1e-4:
                            solutions.append(mid_val)
                        break
        if solutions:
            break
    return solutions


# Keep old find_next_handle for backward compatibility
def find_next_handle(theta_prev, target_d, a=None, direction='inward'):
    """Simple wrapper: find the closest solution in preferred direction."""
    sols = find_all_solutions(theta_prev, target_d, a)
    if not sols:
        return None
    if direction == 'inward':
        inward = [s for s in sols if s < theta_prev]
        return max(inward) if inward else min(sols, key=lambda s: abs(s - theta_prev))
    elif direction == 'outward':
        outward = [s for s in sols if s > theta_prev]
        return min(outward) if outward else min(sols, key=lambda s: abs(s - theta_prev))
    return min(sols, key=lambda s: abs(s - theta_prev))

# ============================================================
# Board vertex computation (for collision detection)
# ============================================================
def get_board_vertices(theta_front, theta_rear, a=None):
    """
    Compute the four vertices of a board given its front and rear handle thetas.

    The board has width 0.30m, handles are 0.275m from each end.
    The board's centerline connects the two handle holes.

    Returns:
        front_inner, front_outer, rear_inner, rear_outer (each (x, y))
        where inner = toward spiral center, outer = away from center
    """
    if a is None:
        a = A_COEF

    x_f, y_f = spiral_to_xy(theta_front, a)
    x_r, y_r = spiral_to_xy(theta_rear, a)

    # Board direction vector (front to rear)
    dx = x_r - x_f
    dy = y_r - y_f
    length = np.sqrt(dx**2 + dy**2)
    if length < 1e-12:
        return None

    # Unit vectors
    ux, uy = dx / length, dy / length  # along board
    nx, ny = -uy, ux                   # perpendicular (one side)

    # Board extends HOLE_OFFSET beyond handles at each end
    # Front vertex position = front handle - HOLE_OFFSET * u
    # Rear vertex position = rear handle + HOLE_OFFSET * u

    hw = HALF_WIDTH  # 0.15m

    # Inner/outer depends on which side faces the spiral center
    # For clockwise spiral, the "inner" side is toward decreasing radius
    # We compute both sides and determine inner/outer based on position
    # relative to the spiral center

    # Four vertices (without distinguishing inner/outer yet):
    # Front end: two vertices perpendicular to board direction
    front_center = np.array([x_f - HOLE_OFFSET * ux, y_f - HOLE_OFFSET * uy])
    rear_center = np.array([x_r + HOLE_OFFSET * ux, y_r + HOLE_OFFSET * uy])

    v1 = front_center + hw * np.array([nx, ny])
    v2 = front_center - hw * np.array([nx, ny])
    v3 = rear_center + hw * np.array([nx, ny])
    v4 = rear_center - hw * np.array([nx, ny])

    # Classify: inner vertices are closer to origin (spiral center)
    d1 = np.linalg.norm(v1)
    d2 = np.linalg.norm(v2)
    d3 = np.linalg.norm(v3)
    d4 = np.linalg.norm(v4)

    # Each end has one inner and one outer vertex
    if d1 < d2:
        front_inner, front_outer = v1, v2
    else:
        front_inner, front_outer = v2, v1

    if d3 < d4:
        rear_inner, rear_outer = v3, v4
    else:
        rear_inner, rear_outer = v4, v3

    return {
        'front_inner': front_inner,
        'front_outer': front_outer,
        'rear_inner': rear_inner,
        'rear_outer': rear_outer,
        'front_center': front_center,
        'rear_center': rear_center,
    }


def point_to_line_distance(point, line_pt1, line_pt2):
    """Distance from a point to a line segment."""
    v = line_pt2 - line_pt1
    w = point - line_pt1
    c1 = np.dot(w, v)
    if c1 <= 0:
        return np.linalg.norm(point - line_pt1)
    c2 = np.dot(v, v)
    if c2 <= c1:
        return np.linalg.norm(point - line_pt2)
    b = c1 / c2
    pb = line_pt1 + b * v
    return np.linalg.norm(point - pb)


def check_collision_at_time(t, a=None):
    """
    Check if collision occurs at time t.

    Per A242 paper: check distance from inner-loop board's outer vertices
    to the adjacent outer-loop board's centerline (full board length).
    Collision if distance < 0.15m (half board width).

    Strategy: for every board i, find boards j on the adjacent outer loop
    (theta_j ≈ theta_i + 2π), then compute vertex-to-centerline distances.

    Returns:
        (has_collision, min_distance, collision_info)
    """
    if a is None:
        a = A_COEF

    theta_array, fail_handle = compute_dragon_at_time(t, a)
    if theta_array is None:
        return True, 0.0, f'geometric infeasibility at handle {fail_handle}'

    # Pre-compute all board vertices (for efficiency)
    all_boards = []
    for i in range(N_HANDLES - 1):
        vi = get_board_vertices(theta_array[i], theta_array[i + 1], a)
        all_boards.append(vi)

    min_dist = float('inf')
    collision_pair = None

    for i in range(N_HANDLES - 1):
        vi = all_boards[i]
        if vi is None:
            continue

        theta_i = theta_array[i]

        for j in range(N_HANDLES - 1):
            # Skip same board, adjacent boards (share a handle), and reversed
            if j == i or j == i + 1 or j + 1 == i:
                continue

            vj = all_boards[j]
            if vj is None:
                continue

            theta_j = theta_array[j]
            dtheta = theta_j - theta_i

            # Only check adjacent spiral loops (one loop = 2π apart)
            # Board at θ+2π is on the adjacent outer loop
            if not (1.5 * np.pi < dtheta < 2.5 * np.pi):
                continue

            # Board j's full centerline: from front end to rear end of the board
            line_j_pt1 = vj['front_center']
            line_j_pt2 = vj['rear_center']

            # Inner board i's outer vertices (facing the outer loop)
            for vname in ['front_outer', 'rear_outer']:
                d = point_to_line_distance(vi[vname], line_j_pt1, line_j_pt2)
                if d < min_dist:
                    min_dist = d
                    collision_pair = (i, j, vname)

            # Board i's full centerline
            line_i_pt1 = vi['front_center']
            line_i_pt2 = vi['rear_center']

            # Outer board j's inner vertices (facing the inner loop)
            for vname in ['front_inner', 'rear_inner']:
                d = point_to_line_distance(vj[vname], line_i_pt1, line_i_pt2)
                if d < min_dist:
                    min_dist = d
                    collision_pair = (i, j, vname + '_reverse')

    has_collision = min_dist < HALF_WIDTH  # 0.15m
    return has_collision, min_dist, collision_pair


# ============================================================
# Simulation (unchanged from v1)
# ============================================================
def simulate_time_range(t_start, t_end, dt=1.0, a=None):
    """Simulate dragon positions for t in [t_start, t_end] with step dt."""
    if a is None:
        a = A_COEF
    n_steps = int((t_end - t_start) / dt) + 1
    t_array = np.linspace(t_start, t_end, n_steps)

    theta_all = np.full((n_steps, N_HANDLES), np.nan)
    failed_at = None

    for idx, t in enumerate(t_array):
        result, fail_handle = compute_dragon_at_time(t, a)
        if result is None:
            if failed_at is None:
                failed_at = (idx, t, fail_handle)
            break
        theta_all[idx, :] = result

    n_valid = np.sum(~np.isnan(theta_all[:, 0]))
    positions = np.full((n_steps, N_HANDLES, 2), np.nan)
    velocities = np.full((n_steps, N_HANDLES), np.nan)

    for idx in range(n_valid):
        for i in range(N_HANDLES):
            x, y = spiral_to_xy(theta_all[idx, i], a)
            positions[idx, i, 0] = x
            positions[idx, i, 1] = y

    for idx in range(n_valid):
        if idx == 0:
            if n_valid >= 2:
                for i in range(N_HANDLES):
                    dx = positions[1, i, 0] - positions[0, i, 0]
                    dy = positions[1, i, 1] - positions[0, i, 1]
                    velocities[0, i] = np.sqrt(dx**2 + dy**2) / dt
        elif idx == n_valid - 1:
            for i in range(N_HANDLES):
                dx = positions[idx, i, 0] - positions[idx - 1, i, 0]
                dy = positions[idx, i, 1] - positions[idx - 1, i, 1]
                velocities[idx, i] = np.sqrt(dx**2 + dy**2) / dt
        else:
            for i in range(N_HANDLES):
                dx = positions[idx + 1, i, 0] - positions[idx - 1, i, 0]
                dy = positions[idx + 1, i, 1] - positions[idx - 1, i, 1]
                velocities[idx, i] = np.sqrt(dx**2 + dy**2) / (2 * dt)

    return t_array, positions, velocities, theta_all, failed_at


# Key handle indices
KEY_HANDLES = {
    '龙头': 0,
    '第1节龙身': 1,
    '第51节龙身': 51,
    '第101节龙身': 101,
    '第151节龙身': 151,
    '第201节龙身': 201,
    '龙尾（后）': N_HANDLES - 1,
}
