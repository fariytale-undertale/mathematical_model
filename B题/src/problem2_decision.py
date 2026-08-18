"""
Problem 2: Production Decision Optimization for 2-Component Assembly
====================================================================
MDP-based decision model with exact expected-profit calculation.

Decision variables (x1, x2, y, z) ∈ {0,1}⁴:
  x1: test component 1?    x2: test component 2?
  y:  test finished product?
  z:  disassemble defective product?

Key recursion: the production cycle involves potential disassembly→reassembly
loops. We use closed-form geometric-series formulas.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from src.config import TABLE1_SCENARIOS
from src.plot_utils import savefig, new_figure, COLORS, OUTPUT_DIR


def expected_profit(scenario, x1, x2, y, z, verbose=False):
    """
    Compute expected profit per unit for given scenario and decisions.

    Parameters
    ----------
    scenario : dict — with keys p1, buy1, test1, p2, buy2, test2, pf, assy, test_f, price, loss_return, dis
    x1, x2, y, z : int — binary decisions

    Returns
    -------
    float — expected profit per unit (price - expected total cost)
    """
    # Unpack scenario
    p1 = scenario['p1']; buy1 = scenario['buy1']; c_test1 = scenario['test1']
    p2 = scenario['p2']; buy2 = scenario['buy2']; c_test2 = scenario['test2']
    pf = scenario['pf']; assy = scenario['assy']; c_test_f = scenario['test_f']
    price = scenario['price']; loss = scenario['loss_return']; c_dis = scenario['dis']

    # Effective component cost (to get one unit to assembly)
    # If tested: geometric series for retries until we get a good one
    if x1:
        C1 = (buy1 + c_test1) / (1 - p1)  # expected cost per good unit
        q1 = 0  # effective defect rate at assembly
    else:
        C1 = buy1
        q1 = p1

    if x2:
        C2 = (buy2 + c_test2) / (1 - p2)
        q2 = 0
    else:
        C2 = buy2
        q2 = p2

    C_comp = C1 + C2  # total component cost per assembly attempt

    # Probability of successful assembly
    P_input_good = (1 - q1) * (1 - q2)
    P_success = P_input_good * (1 - pf)

    if P_success <= 0:
        return -np.inf  # will never succeed

    # Per-attempt recurring costs
    R = assy + y * c_test_f  # assembly + optional testing
    D = (1 - y) * loss       # return loss if untested bad product reaches customer
    P_fail = 1 - P_success

    # --- Expected total cost to deliver one unit ---
    if z == 0:
        # Scrap on failure: start over with new components
        # C = C_comp + R + P_fail * D + P_fail * C
        # => C * P_success = C_comp + R + P_fail * D
        C_total = (C_comp + R + P_fail * D) / P_success

    else:
        # Disassemble on failure:
        # First, determine retry behavior based on whether components are tested
        if x1 and x2:
            # Both tested: components are KNOWN good after disassembly
            # After disassembly: back to assembly, NO new component cost
            # Each retry: R + P_fail * D + P_fail * (dis + retry)
            # C_retry = R + P_fail * D + P_fail * (dis + C_retry)
            # C_retry * (1 - P_fail) = R + P_fail * (D + dis)
            # C_retry = [R + P_fail * (D + dis)] / P_success
            C_retry = (R + P_fail * (D + c_dis)) / P_success
            C_total = C_comp + C_retry

        else:
            # At least one component untested: components may be defective
            # If defective → will NEVER succeed → disassembly creates infinite loop
            # Optimal: try disassembly once, if fails again → scrap
            # (because posterior P(both good | 2 failures) is very low)

            # First attempt
            C_total = C_comp + R + P_fail * D

            # Disassembly + one retry
            C_retry_one = c_dis + R + P_fail * D  # cost of retry attempt (no comp cost)
            P_success_retry = P_success  # same probability (same components)

            # After retry failure: scrap (buy new components)
            C_after_retry_fail = C_comp + R + P_fail * D + P_fail * C_comp / P_success
            # Wait, this is still recursive. Let me handle it properly.

            # After retry, if fails again: scrap and restart
            # P(retry succeeds) = P_success
            # If retry fails: scrap cost = retry_cost + C_total (start over)
            # But C_total includes first attempt + retry...
            # Let me think again.

            # Decision tree:
            # 1. First attempt: cost C_comp + R + P_fail*D
            #    - Success (P_success): done, net = -(above) from profit perspective
            #    - Fail (P_fail): go to step 2
            # 2. Disassemble + retry: cost c_dis + R + P_fail*D
            #    - Success (P_success): done
            #    - Fail (P_fail): scrap, start over → add C_total

            # This gives a recursive equation:
            # C_total = C_comp + R + P_fail*D + P_fail * [c_dis + R + P_fail*D + P_fail * C_total]
            # C_total = C_first + P_fail * [C_retry + P_fail * C_total]
            # where C_first = C_comp + R + P_fail*D
            #       C_retry = c_dis + R + P_fail*D

            # C_total - P_fail^2 * C_total = C_first + P_fail * C_retry
            # C_total * (1 - P_fail^2) = C_first + P_fail * C_retry

            C_first = C_comp + R + P_fail * D
            C_retry = c_dis + R + P_fail * D
            denom = 1 - P_fail**2
            if denom <= 0:
                C_total = np.inf
            else:
                C_total = (C_first + P_fail * C_retry) / denom

    profit = price - C_total

    if verbose:
        print(f"  Decisions: x1={x1}, x2={x2}, y={y}, z={z}")
        print(f"    C_comp={C_comp:.2f}, P_success={P_success:.4f}")
        print(f"    C_total={C_total:.2f}, profit={profit:.2f}")

    return profit


def solve_scenario(scenario, scenario_idx):
    """Find optimal decisions for one scenario."""
    best_profit = -np.inf
    best_decisions = None
    results = []

    for x1, x2, y, z in product([0, 1], repeat=4):
        profit = expected_profit(scenario, x1, x2, y, z)
        results.append({
            'x1': x1, 'x2': x2, 'y': y, 'z': z,
            'profit': profit
        })
        if profit > best_profit:
            best_profit = profit
            best_decisions = (x1, x2, y, z)

    # Also compute "do nothing" baseline (no testing, no disassembly)
    baseline = expected_profit(scenario, 0, 0, 0, 0)

    return best_decisions, best_profit, baseline, results


def solve_problem2():
    """Solve Problem 2 for all 6 scenarios."""
    print("=" * 60)
    print("问题2: 两零配件生产过程决策优化")
    print("=" * 60)

    all_results = []

    for idx, scenario in enumerate(TABLE1_SCENARIOS):
        best_dec, best_profit, baseline, details = solve_scenario(scenario, idx + 1)
        x1, x2, y, z = best_dec

        # Compute key intermediate values
        p1, p2, pf = scenario['p1'], scenario['p2'], scenario['pf']
        if x1:
            C1 = (scenario['buy1'] + scenario['test1']) / (1 - p1)
            q1 = 0
        else:
            C1 = scenario['buy1']
            q1 = p1
        if x2:
            C2 = (scenario['buy2'] + scenario['test2']) / (1 - p2)
            q2 = 0
        else:
            C2 = scenario['buy2']
            q2 = p2
        P_success = (1 - q1) * (1 - q2) * (1 - pf)

        result = {
            '情形': idx + 1,
            '检测零配件1': '是' if x1 else '否',
            '检测零配件2': '是' if x2 else '否',
            '检测成品': '是' if y else '否',
            '拆解不合格品': '是' if z else '否',
            '期望利润(元/件)': f'{best_profit:.2f}',
            '基线利润(元/件)': f'{baseline:.2f}',
            '利润提升(元/件)': f'{best_profit - baseline:.2f}',
            '成品合格率': f'{P_success:.4f}',
        }

        all_results.append(result)

        print(f"\n情形 {idx+1}:")
        print(f"  最优决策: 检测零配件1={'是' if x1 else '否'}, "
              f"检测零配件2={'是' if x2 else '否'}, "
              f"检测成品={'是' if y else '否'}, "
              f"拆解={'是' if z else '否'}")
        print(f"  期望利润: {best_profit:.2f} 元/件 (基线: {baseline:.2f})")
        print(f"  成品合格率: {P_success:.4f}")

    # --- Display results table ---
    df = pd.DataFrame(all_results)
    print("\n" + "=" * 60)
    print("决策方案汇总表:")
    print(df.to_string(index=False))

    # --- Sensitivity analysis for Scenario 1 ---
    print("\n--- 敏感性分析 (情形1) ---")
    sc1 = TABLE1_SCENARIOS[0]

    # Vary pf (finished product defect rate)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('情形1 参数敏感性分析', fontsize=14, fontweight='bold')

    # (a) Vary pf
    pf_range = np.linspace(0.02, 0.30, 30)
    profits_pf = {d: [] for d in [(0,0,0,0), (0,0,0,1), (0,0,1,0), (1,0,0,0),
                                    (1,1,0,0), (1,1,0,1), (1,1,1,0), (1,1,1,1)]}
    for pf_v in pf_range:
        sc = dict(sc1); sc['pf'] = pf_v
        for dec in profits_pf:
            profits_pf[dec].append(expected_profit(sc, *dec))

    ax = axes[0, 0]
    for dec, color in zip(profits_pf, [COLORS[i % len(COLORS)] for i in range(len(profits_pf))]):
        label = f"({dec[0]},{dec[1]},{dec[2]},{dec[3]})"
        ax.plot(pf_range, profits_pf[dec], color=color, linewidth=1.2, alpha=0.7, label=label)
    ax.set_xlabel('成品次品率 p_f')
    ax.set_ylabel('期望利润 (元/件)')
    ax.set_title('(a) 成品次品率影响')
    ax.legend(fontsize=6, ncol=2, loc='lower left')
    ax.grid(True, alpha=0.3)

    # (b) Vary p1
    p1_range = np.linspace(0.02, 0.35, 30)
    profits_p1 = {d: [] for d in profits_pf}
    for p1_v in p1_range:
        sc = dict(sc1); sc['p1'] = p1_v
        for dec in profits_p1:
            profits_p1[dec].append(expected_profit(sc, *dec))

    ax = axes[0, 1]
    for dec, color in zip(profits_p1, [COLORS[i % len(COLORS)] for i in range(len(profits_p1))]):
        ax.plot(p1_range, profits_p1[dec], color=color, linewidth=1.2, alpha=0.7)
    ax.set_xlabel('零配件1次品率 p1')
    ax.set_ylabel('期望利润 (元/件)')
    ax.set_title('(b) 零配件1次品率影响')
    ax.grid(True, alpha=0.3)

    # (c) Vary disassembly cost
    dis_range = np.linspace(1, 30, 30)
    profits_dis = {d: [] for d in profits_pf}
    for dis_v in dis_range:
        sc = dict(sc1); sc['dis'] = dis_v
        for dec in profits_dis:
            profits_dis[dec].append(expected_profit(sc, *dec))

    ax = axes[1, 0]
    for dec, color in zip(profits_dis, [COLORS[i % len(COLORS)] for i in range(len(profits_dis))]):
        ax.plot(dis_range, profits_dis[dec], color=color, linewidth=1.2, alpha=0.7)
    ax.set_xlabel('拆解费用 (元/件)')
    ax.set_ylabel('期望利润 (元/件)')
    ax.set_title('(c) 拆解费用影响')
    ax.grid(True, alpha=0.3)

    # (d) Vary loss_return
    loss_range = np.linspace(2, 50, 30)
    profits_loss = {d: [] for d in profits_pf}
    for loss_v in loss_range:
        sc = dict(sc1); sc['loss_return'] = loss_v
        for dec in profits_loss:
            profits_loss[dec].append(expected_profit(sc, *dec))

    ax = axes[1, 1]
    for dec, color in zip(profits_loss, [COLORS[i % len(COLORS)] for i in range(len(profits_loss))]):
        ax.plot(loss_range, profits_loss[dec], color=color, linewidth=1.2, alpha=0.7)
    ax.set_xlabel('调换损失 (元/件)')
    ax.set_ylabel('期望利润 (元/件)')
    ax.set_title('(d) 调换损失影响')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig(fig, 'problem2_sensitivity.png')
    plt.close(fig)

    # --- Profit comparison bar chart ---
    fig, ax = new_figure((12, 5), title='各情形最优决策 vs 基线 期望利润对比')
    x_pos = np.arange(len(all_results))
    best_profits = [float(r['期望利润(元/件)']) for r in all_results]
    baseline_profits = [float(r['基线利润(元/件)']) for r in all_results]
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, best_profits, width, label='最优决策', color=COLORS[0], alpha=0.9)
    bars2 = ax.bar(x_pos + width/2, baseline_profits, width, label='基线(全不检测不拆解)', color=COLORS[1], alpha=0.7)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', fontsize=8)
    ax.set_xlabel('情形')
    ax.set_ylabel('期望利润 (元/件)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'情形{i+1}' for i in range(len(all_results))])
    ax.legend()
    savefig(fig, 'problem2_profit_comparison.png')
    plt.close(fig)

    # --- Top-5 decisions per scenario ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('各情形 Top-5 决策方案利润对比', fontsize=14, fontweight='bold')
    for s_idx in range(6):
        ax = axes[s_idx // 3, s_idx % 3]
        sc = TABLE1_SCENARIOS[s_idx]
        all_dec = []
        for x1, x2, y, z in product([0, 1], repeat=4):
            profit = expected_profit(sc, x1, x2, y, z)
            all_dec.append((profit, (x1, x2, y, z)))
        all_dec.sort(key=lambda x: x[0], reverse=True)
        top5 = all_dec[:5]
        labels = [f"({d[0]},{d[1]},{d[2]},{d[3]})" for _, d in top5]
        profits = [p for p, _ in top5]
        colors_bar = [COLORS[0] if i == 0 else COLORS[i % len(COLORS)] for i in range(5)]
        ax.barh(range(5), profits, color=colors_bar, alpha=0.85, edgecolor='white')
        ax.set_yticks(range(5))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f'情形 {s_idx+1}')
        ax.set_xlabel('期望利润 (元/件)')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2, axis='x')
    plt.tight_layout()
    savefig(fig, 'problem2_top5_decisions.png')
    plt.close(fig)

    # --- Cost breakdown for optimal decision of each scenario ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('各情形最优决策的成本构成分析', fontsize=14, fontweight='bold')
    cost_labels = ['零配件采购', '零配件检测', '装配', '成品检测', '拆解', '调换损失']
    cost_colors = ['#2166AC', '#D6604D', '#4DAF4A', '#984EA3', '#FF7F00', '#A65628']
    for s_idx in range(6):
        ax = axes[s_idx // 3, s_idx % 3]
        sc = TABLE1_SCENARIOS[s_idx]
        best_dec, _, _, details = solve_scenario(sc, s_idx + 1)
        x1, x2, y, z = best_dec
        # Compute individual cost components for one unit attempt
        C1_eff = (sc['buy1'] + sc['test1']) / (1 - sc['p1']) if x1 else sc['buy1']
        C2_eff = (sc['buy2'] + sc['test2']) / (1 - sc['p2']) if x2 else sc['buy2']
        P_success = (1 - (0 if x1 else sc['p1'])) * (1 - (0 if x2 else sc['p2'])) * (1 - sc['pf'])
        P_fail = 1 - P_success
        # Approximate cost breakdown (using z=1 or z=0 formula)
        c_comp = C1_eff + C2_eff
        c_test_comp = (sc['test1'] / (1 - sc['p1']) if x1 else 0) + (sc['test2'] / (1 - sc['p2']) if x2 else 0)
        c_assy = sc['assy']
        c_test_f = sc['test_f'] if y else 0
        # Expected per-successful-unit costs
        if P_success > 0:
            scale = 1.0 / P_success if z == 0 else (1.0 + P_fail / P_success)
            costs = [
                (C1_eff + C2_eff - c_test_comp) * scale,
                c_test_comp * scale,
                c_assy * scale,
                c_test_f * scale,
                sc['dis'] * P_fail * scale if z else 0,
                sc['loss_return'] * (1 - y) * P_fail * scale,
            ]
        else:
            costs = [0] * 6
        # Filter zero costs
        nonzero = [(l, c) for l, c in zip(cost_labels, costs) if c > 0.01]
        if nonzero:
            lbls, vals = zip(*nonzero)
            wedges, texts, autotexts = ax.pie(vals, labels=lbls, colors=cost_colors[:len(lbls)],
                                                autopct='%1.1f%%', textprops={'fontsize': 8})
        ax.set_title(f'情形 {s_idx+1} (利润={expected_profit(sc, *best_dec):.1f})', fontsize=11)
    plt.tight_layout()
    savefig(fig, 'problem2_cost_breakdown.png')
    plt.close(fig)

    # --- Contour: p1 vs pf joint sensitivity (Scenario 1) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sc1 = TABLE1_SCENARIOS[0]
    p1_grid = np.linspace(0.03, 0.30, 40)
    pf_grid = np.linspace(0.03, 0.30, 40)
    P1, PF = np.meshgrid(p1_grid, pf_grid)
    # Find optimal decision at each point
    Z_profit = np.zeros_like(P1)
    Z_decision = np.zeros_like(P1, dtype=int)
    for i in range(len(pf_grid)):
        for j in range(len(p1_grid)):
            sc = dict(sc1); sc['p1'] = P1[i, j]; sc['pf'] = PF[i, j]
            best = -np.inf; best_idx = 0
            for idx_d, (xx1, xx2, yy, zz) in enumerate(product([0, 1], repeat=4)):
                p = expected_profit(sc, xx1, xx2, yy, zz)
                if p > best:
                    best = p; best_idx = idx_d
            Z_profit[i, j] = best
            Z_decision[i, j] = best_idx

    # Profit contour
    ax = axes[0]
    cs = ax.contourf(P1, PF, Z_profit, levels=20, cmap='RdYlBu', alpha=0.9)
    ax.contour(P1, PF, Z_profit, levels=8, colors='black', linewidths=0.5, alpha=0.4)
    plt.colorbar(cs, ax=ax, label='期望利润 (元/件)')
    ax.set_xlabel('零配件1次品率 p1'); ax.set_ylabel('成品次品率 pf')
    ax.set_title('(a) 期望利润等高线 (p1 vs pf)')

    # Decision region
    ax = axes[1]
    cs2 = ax.contourf(P1, PF, Z_decision, levels=np.arange(-0.5, 16.5, 1), cmap='tab20', alpha=0.85)
    ax.set_xlabel('零配件1次品率 p1'); ax.set_ylabel('成品次品率 pf')
    ax.set_title('(b) 最优决策区域 (不同颜色=不同决策)')
    plt.tight_layout()
    savefig(fig, 'problem2_contour_sensitivity.png')
    plt.close(fig)

    return df, all_results


if __name__ == '__main__':
    solve_problem2()
