"""
Problem 3: Generalized Assembly Tree Decision Optimization
===========================================================
For m processes and n components with a tree/DAG assembly structure.

Method: Bottom-up DP on the assembly DAG.
- Leaf nodes: components (purchase, test-or-not)
- Internal nodes: subassemblies (assemble, test-or-not, disassemble-or-not)
- Root: final product → market

The DP recursively computes the optimal policy for each subtree,
propagating effective costs and defect rates upward.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from collections import defaultdict
from src.config import TABLE2_COMPONENTS, TABLE2_SEMI, TABLE2_FINAL
from src.plot_utils import savefig, new_figure, COLORS, OUTPUT_DIR


class AssemblyNode:
    """A node in the assembly tree (component, semi-product, or final product)."""

    def __init__(self, node_id, node_type='component'):
        """
        Parameters
        ----------
        node_id : int or str
        node_type : str — 'component', 'semi', or 'final'
        """
        self.id = node_id
        self.type = node_type
        self.children = []       # list of AssemblyNode (inputs)
        self.parent = None

        # Parameters (populated from config)
        self.p_defect = 0.0      # defect rate (for this assembly step)
        self.c_buy = 0.0         # purchase cost (components only)
        self.c_test = 0.0        # testing cost
        self.c_assy = 0.0        # assembly cost (non-leaf only)
        self.c_dis = 0.0         # disassembly cost (non-leaf only)
        self.price = 0.0         # market price (final only)
        self.c_loss = 0.0        # return loss (final only)

        # Computed optimal policy
        self.do_test = False
        self.do_disassemble = False
        self.effective_cost = 0.0     # effective cost to produce one unit
        self.effective_defect = 0.0   # effective defect rate after decisions


def build_assembly_tree():
    """Build the assembly tree from Table 2 configuration."""
    # Create component nodes
    components = []
    for idx, p, buy, test_cost in TABLE2_COMPONENTS:
        node = AssemblyNode(f'C{idx}', 'component')
        node.p_defect = p
        node.c_buy = buy
        node.c_test = test_cost
        components.append(node)

    # Create semi-product nodes
    semi_products = []
    semi_idx = 0
    for _, pf, assy, test_cost, dis_cost, child_indices in TABLE2_SEMI:
        node = AssemblyNode(f'S{semi_idx+1}', 'semi')
        semi_idx += 1
        node.p_defect = pf
        node.c_assy = assy
        node.c_test = test_cost
        node.c_dis = dis_cost
        for ci in child_indices:
            node.children.append(components[ci])
            components[ci].parent = node
        semi_products.append(node)

    # Create final product node
    final = AssemblyNode('F', 'final')
    final.p_defect = TABLE2_FINAL['pf']
    final.c_assy = TABLE2_FINAL['assy']
    final.c_test = TABLE2_FINAL['test']
    final.c_dis = TABLE2_FINAL['dis']
    final.price = TABLE2_FINAL['price']
    final.c_loss = TABLE2_FINAL['loss_return']
    for si in TABLE2_FINAL['input_semi']:
        final.children.append(semi_products[si])
        semi_products[si].parent = final

    return components, semi_products, final


def compute_leaf_policy(component):
    """
    Compute optimal policy for a leaf component node.
    Returns (effective_cost, effective_defect_rate).
    """
    p = component.p_defect
    buy = component.c_buy
    test_c = component.c_test

    # Option 1: Don't test
    cost_no_test = buy
    defect_no_test = p

    # Option 2: Test (discard bad ones, buy new)
    if p >= 1.0:
        cost_test = np.inf  # can never get a good one
        defect_test = 0.0
    else:
        cost_test = (buy + test_c) / (1 - p)
        defect_test = 0.0  # all bad discarded

    # Choose minimum cost
    if cost_test < cost_no_test:
        component.do_test = True
        component.effective_cost = cost_test
        component.effective_defect = defect_test
    else:
        component.do_test = False
        component.effective_cost = cost_no_test
        component.effective_defect = defect_no_test

    return component.effective_cost, component.effective_defect


def compute_node_policy(node):
    """
    Compute optimal policy for a non-leaf node (semi or final product).

    The node receives inputs from children (components or semi-products).
    Each child has already been optimized (effective_cost, effective_defect).

    We need to decide:
    - Test the assembled product? (y)
    - Disassemble defective products? (z)
    """
    # Aggregate input costs and defect rates
    total_input_cost = sum(c.effective_cost for c in node.children)
    # P(all inputs good) = prod(1 - defect_i)
    p_all_inputs_good = np.prod([1 - c.effective_defect for c in node.children])

    if p_all_inputs_good <= 0:
        return np.inf, 1.0  # impossible to get good inputs

    pf = node.p_defect
    P_success = p_all_inputs_good * (1 - pf)
    P_fail = 1 - P_success

    assy = node.c_assy
    test_f = node.c_test
    dis = node.c_dis

    # Note: For non-final nodes, there's no "customer return" — products are
    # passed to the next assembly stage. Defective products are detected
    # by testing at THIS node (if testing is done).
    # If untested, defective products flow to the parent → cause defects there.

    # For final product: untested defective products reach customer, cause loss
    if node.type == 'final':
        loss = node.c_loss
    else:
        loss = 0  # no customer return for intermediate products

    best_profit = -np.inf  # actually we minimize cost here
    best_cost = np.inf
    best_policy = (0, 0)   # (y, z)

    for y, z in product([0, 1], repeat=2):
        # Compute expected total cost for this node (per good output unit)
        R = assy + y * test_f
        D = (1 - y) * loss

        if z == 0:
            # Scrap on failure
            if P_success <= 0:
                cost = np.inf
            else:
                cost = (total_input_cost + R + P_fail * D) / P_success

        else:
            # Disassemble on failure
            # Components are recovered, back to assembly with same quality
            # For "tested" children (defect=0): known good → retry helps
            # For "untested" children: same quality → may be defective → retry fails

            # Check if ALL children are tested (known good)
            all_tested = all(c.effective_defect == 0 for c in node.children)

            if all_tested:
                # Disassembled components are KNOWN good
                # C_retry = R + P_fail * D + P_fail * (dis + C_retry)
                # C_retry * P_success = R + P_fail * (D + dis)
                if P_success <= 0:
                    cost = np.inf
                else:
                    C_retry = (R + P_fail * (D + dis)) / P_success
                    cost = total_input_cost + C_retry
            else:
                # At least one child untested → retry with same quality
                # Try disassembly once; if fails again → scrap
                C_first = total_input_cost + R + P_fail * D
                C_retry = dis + R + P_fail * D  # no new component cost
                denom = 1 - P_fail**2
                if denom <= 0:
                    cost = np.inf
                else:
                    cost = (C_first + P_fail * C_retry) / denom

        if cost < best_cost:
            best_cost = cost
            best_policy = (y, z)

    y_opt, z_opt = best_policy
    if node.type == 'final':
        node.effective_cost = best_cost
    else:
        node.effective_cost = best_cost

    # Effective defect rate: if testing, bad ones are caught
    if y_opt:
        node.effective_defect = 0  # tested, only good pass through
    else:
        node.effective_defect = P_fail  # bad products flow through

    return best_cost, y_opt, z_opt


def evaluate_full_tree(comp_test_vec, components, semi_products, final):
    """
    Evaluate the full assembly tree given component testing decisions.
    For each semi-product and final product, find optimal (y,z).

    Returns (total_profit, comp_decisions, semi_decisions, final_decision).
    """
    # Apply component testing decisions
    for i, comp in enumerate(components):
        comp.do_test = bool(comp_test_vec[i])
        if comp.do_test:
            if comp.p_defect >= 1.0:
                comp.effective_cost = np.inf
                comp.effective_defect = 0.0
            else:
                comp.effective_cost = (comp.c_buy + comp.c_test) / (1 - comp.p_defect)
                comp.effective_defect = 0.0
        else:
            comp.effective_cost = comp.c_buy
            comp.effective_defect = comp.p_defect

    # Check for infeasible components
    if any(np.isinf(c.effective_cost) for c in components):
        return -np.inf, None, None, None

    # Optimize each semi-product
    semi_best = []
    for semi in semi_products:
        best_semi_cost = np.inf
        best_semi_yz = (0, 0)
        for y, z in product([0, 1], repeat=2):
            # Temporarily set children's effective values (already set above)
            cost, _, _ = compute_node_policy_with_decision(semi, y, z)
            if cost < best_semi_cost:
                best_semi_cost = cost
                best_semi_yz = (y, z)

        # Apply best decision
        y_opt, z_opt = best_semi_yz
        cost_opt, _, _ = compute_node_policy_with_decision(semi, y_opt, z_opt)
        semi.effective_cost = cost_opt
        semi_best.append((y_opt, z_opt, cost_opt))

    # Optimize final product
    best_final_profit = -np.inf
    best_final_yz = (0, 0)
    best_final_cost = np.inf
    for y, z in product([0, 1], repeat=2):
        cost, _, _ = compute_node_policy_with_decision(final, y, z)
        profit = final.price - cost
        if profit > best_final_profit:
            best_final_profit = profit
            best_final_yz = (y, z)
            best_final_cost = cost

    # Store final effective cost for downstream use (e.g., cost breakdown)
    final.effective_cost = best_final_cost

    # Build result dicts
    comp_dec = [{'零配件': c.id, '检测': '是' if c.do_test else '否',
                 '有效成本': f'{c.effective_cost:.2f}',
                 '有效次品率': f'{c.effective_defect:.4f}'} for c in components]
    semi_dec = [{'半成品': s.id, '检测': '是' if yz[0] else '否',
                 '拆解': '是' if yz[1] else '否',
                 '有效成本': f'{cost:.2f}'}
                for s, (yz, cost) in zip(semi_products,
                                         [(semi_best[i][:2], semi_best[i][2]) for i in range(len(semi_products))])]
    # Wait, semi_best stores (y, z, cost). Let me fix this.
    semi_dec_corrected = []
    for i, semi in enumerate(semi_products):
        yz, cost = (semi_best[i][0], semi_best[i][1]), semi_best[i][2]
        semi_dec_corrected.append({
            '半成品': semi.id,
            '检测': '是' if yz[0] else '否',
            '拆解': '是' if yz[1] else '否',
            '有效成本': f'{cost:.2f}',
            '有效次品率': f'{semi.effective_defect:.4f}',
        })

    final_dec = {'成品': 'F', '检测': '是' if best_final_yz[0] else '否',
                 '拆解': '是' if best_final_yz[1] else '否',
                 '总成本': f'{best_final_cost:.2f}',
                 '售价': f'{final.price:.2f}',
                 '期望利润': f'{best_final_profit:.2f}'}

    return best_final_profit, comp_dec, semi_dec_corrected, final_dec


def compute_node_policy_with_decision(node, y, z):
    """
    Compute effective cost for a node given specific (y, z) decisions.
    Returns (effective_cost, y, z).
    Does NOT mutate node state (effective_defect set temporarily).
    """
    # Aggregate input costs and defect rates
    total_input_cost = sum(c.effective_cost for c in node.children)
    p_all_inputs_good = np.prod([1 - c.effective_defect for c in node.children])

    if p_all_inputs_good <= 0:
        return np.inf, y, z

    pf = node.p_defect
    P_success = p_all_inputs_good * (1 - pf)
    P_fail = 1 - P_success

    assy = node.c_assy
    test_f = node.c_test
    dis = node.c_dis
    loss = node.c_loss if node.type == 'final' else 0

    R = assy + y * test_f
    D = (1 - y) * loss

    if z == 0:
        if P_success <= 0:
            cost = np.inf
        else:
            cost = (total_input_cost + R + P_fail * D) / P_success
    else:
        all_tested = all(c.effective_defect == 0 for c in node.children)
        if all_tested:
            if P_success <= 0:
                cost = np.inf
            else:
                C_retry = (R + P_fail * (D + dis)) / P_success
                cost = total_input_cost + C_retry
        else:
            C_first = total_input_cost + R + P_fail * D
            C_retry = dis + R + P_fail * D
            denom = 1 - P_fail**2
            if denom <= 0:
                cost = np.inf
            else:
                cost = (C_first + P_fail * C_retry) / denom

    return cost, y, z


def solve_assembly_tree():
    """Optimize the assembly tree using enumeration over component testing decisions."""
    components, semi_products, final = build_assembly_tree()

    n_components = len(components)
    best_overall_profit = -np.inf
    best_overall = None

    print(f"  枚举 {2**n_components} 种零配件检测组合...")

    for test_mask in range(2**n_components):
        comp_test_vec = [(test_mask >> i) & 1 for i in range(n_components)]
        profit, comp_dec, semi_dec, final_dec = evaluate_full_tree(
            comp_test_vec, components, semi_products, final)

        if profit > best_overall_profit:
            best_overall_profit = profit
            best_overall = (comp_test_vec, comp_dec, semi_dec, final_dec, profit)

    comp_test_vec, comp_dec, semi_dec, final_dec, profit = best_overall

    print("--- 最优零配件决策 ---")
    for d in comp_dec:
        print(f"  {d['零配件']}: 检测={d['检测']}, 有效成本={d['有效成本']}, 有效次品率={d['有效次品率']}")

    print("\n--- 最优半成品决策 ---")
    for d in semi_dec:
        print(f"  {d['半成品']}: 检测={d['检测']}, 拆解={d['拆解']}, 有效成本={d['有效成本']}, 有效次品率={d['有效次品率']}")

    print(f"\n--- 最优成品决策 ---")
    print(f"  成品: 检测={final_dec['检测']}, 拆解={final_dec['拆解']}, "
          f"总成本={final_dec['总成本']}, 期望利润={final_dec['期望利润']}")

    # Restore best decisions to the objects for visualization
    evaluate_full_tree(comp_test_vec, components, semi_products, final)

    return comp_dec, semi_dec, final_dec, components, semi_products, final, profit


def solve_problem3():
    """Solve Problem 3."""
    print("=" * 60)
    print("问题3: m道工序n个零配件 — 装配树决策优化")
    print("=" * 60)
    print("\n装配结构: 8个零配件 → 3个半成品 → 1个成品")
    print("  零配件1,2,3 → 半成品1")
    print("  零配件4,5,6 → 半成品2")
    print("  零配件7,8   → 半成品3")
    print("  半成品1,2,3 → 成品\n")

    comp_dec, semi_dec, final_dec, comps, semis, final, profit = solve_assembly_tree()

    # --- Display summary tables ---
    print("\n" + "=" * 60)
    print("零配件决策汇总:")
    print(pd.DataFrame(comp_dec).to_string(index=False))

    print("\n半成品决策汇总:")
    print(pd.DataFrame(semi_dec).to_string(index=False))

    print(f"\n成品决策: {final_dec}")

    # --- Visualize the assembly tree ---
    fig, ax = new_figure((14, 7), title='装配树结构与最优决策')

    # Node positions (manual layout)
    # Components at bottom
    comp_pos = {}
    for i in range(8):
        comp_pos[f'C{i+1}'] = (i * 1.5, 0)

    # Semi-products in middle
    semi_pos = {
        'S1': (1.5, 2),   # avg of C1-C3
        'S2': (6.0, 2),   # avg of C4-C6
        'S3': (9.75, 2),  # avg of C7-C8
    }

    # Final product at top
    final_pos = {'F': (5.25, 4)}

    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 5)
    ax.axis('off')

    # Draw edges
    # C1-C3 → S1
    for i in [1, 2, 3]:
        ax.plot([comp_pos[f'C{i}'][0], semi_pos['S1'][0]],
                [comp_pos[f'C{i}'][1], semi_pos['S1'][1]],
                'k-', alpha=0.3, linewidth=1)
    # C4-C6 → S2
    for i in [4, 5, 6]:
        ax.plot([comp_pos[f'C{i}'][0], semi_pos['S2'][0]],
                [comp_pos[f'C{i}'][1], semi_pos['S2'][1]],
                'k-', alpha=0.3, linewidth=1)
    # C7-C8 → S3
    for i in [7, 8]:
        ax.plot([comp_pos[f'C{i}'][0], semi_pos['S3'][0]],
                [comp_pos[f'C{i}'][1], semi_pos['S3'][1]],
                'k-', alpha=0.3, linewidth=1)
    # S1, S2, S3 → F
    for s in ['S1', 'S2', 'S3']:
        ax.plot([semi_pos[s][0], final_pos['F'][0]],
                [semi_pos[s][1], final_pos['F'][1]],
                'k-', alpha=0.3, linewidth=1)

    # Draw component nodes
    for comp in comps:
        x, y = comp_pos[comp.id]
        color = COLORS[0] if comp.do_test else COLORS[1]
        circle = plt.Circle((x, y), 0.4, color=color, alpha=0.8, ec='black', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(x, y, comp.id, ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        status = 'Y' if comp.do_test else 'N'
        ax.text(x, y - 0.7, f"Test={status}\n{comp.effective_cost:.1f} yuan",
                ha='center', fontsize=7, color=color)

    # Draw semi-product nodes
    for semi in semis:
        x, y = semi_pos[semi.id]
        color = COLORS[3] if semi.effective_defect == 0 else COLORS[4]
        rect = plt.Rectangle((x - 0.6, y - 0.35), 1.2, 0.7, color=color, alpha=0.8, ec='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, semi.id, ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Draw final product node
    x, y = final_pos['F']
    rect = plt.Rectangle((x - 0.7, y - 0.4), 1.4, 0.8, color=COLORS[0], alpha=0.9, ec='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y, f"成品\n{profit:.1f}元/件", ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    ax.set_title('装配树结构与最优决策\n(绿色=检测, 红色=不检测)', fontsize=13, fontweight='bold')
    savefig(fig, 'problem3_assembly_tree.png')
    plt.close(fig)

    # --- Bar chart: component costs ---
    fig, ax = new_figure((10, 5), title='各零配件有效成本 vs 原始购买价')
    x_pos = np.arange(8)
    buy_costs = [c.c_buy for c in comps]
    eff_costs = [c.effective_cost for c in comps]
    width = 0.35
    ax.bar(x_pos - width/2, buy_costs, width, label='原始购买价', color=COLORS[1], alpha=0.7)
    ax.bar(x_pos + width/2, eff_costs, width, label='有效成本(含检测)', color=COLORS[0], alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'C{i+1}' for i in range(8)])
    ax.set_ylabel('成本 (元/件)')
    ax.legend()
    savefig(fig, 'problem3_component_costs.png')
    plt.close(fig)

    # --- Strategy comparison: Optimal vs All-Test vs No-Test ---
    components2, semi_products2, final2 = build_assembly_tree()
    strategies = {
        '最优(选择性检测)': None,  # computed below
        '全检测': [1]*8,
        '全不检测': [0]*8,
        '仅检测低价(C1,4,7)': [1,0,0,1,0,0,1,0],
        '检测低价+中价(C1,2,4,5,7)': [1,1,0,1,1,0,1,0],
    }

    strategy_results = {}
    for name, vec in strategies.items():
        if name == '最优(选择性检测)':
            profit_val = profit  # from optimal solution
        else:
            comps2, semis2, fin2 = build_assembly_tree()
            p, _, _, _ = evaluate_full_tree(vec, comps2, semis2, fin2)
            profit_val = p
        strategy_results[name] = profit_val

    fig, ax = new_figure((10, 5), title='问题3: 不同检测策略的期望利润对比')
    names = list(strategy_results.keys())
    profits_list = list(strategy_results.values())
    colors_bar = [COLORS[0] if '最优' in n else COLORS[i % len(COLORS)] for i, n in enumerate(names)]
    bars = ax.bar(range(len(names)), profits_list, color=colors_bar, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, profits_list):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('期望利润 (元/件)')
    ax.grid(True, alpha=0.2, axis='y')
    savefig(fig, 'problem3_strategy_comparison.png')
    plt.close(fig)

    # --- Cost breakdown for Problem 3 ---
    fig, ax = new_figure((10, 5), title='问题3最优决策的成本构成')
    # Component effective costs (procurement + testing, with geometric retry)
    c_comps = sum(c.effective_cost for c in comps)
    # Assembly costs (semi-products + final, nominal per-attempt)
    c_assy_total = sum(s.c_assy for s in semis) + final.c_assy
    # Testing costs for semi and final products
    c_test_semi_final = sum(s.c_test for s in semis if s.effective_defect == 0) + \
                        (final.c_test if hasattr(final, 'do_test') and final.do_test else 0)
    # The effective cost already accounts for geometric retry (1/P_success)
    # Remaining cost is due to disassembly, return losses, and retry amplification
    total_effective = final.effective_cost
    c_retry_overhead = total_effective - c_comps - c_assy_total - c_test_semi_final
    # Split retry overhead into disassembly and return loss proportionally
    # based on their relative magnitudes in the cost structure
    if c_retry_overhead > 0:
        # Estimate: disassembly cost scales with expected retries, return loss with (1-y)
        ratio_dis = final.c_dis / (final.c_dis + final.c_loss) if (final.c_dis + final.c_loss) > 0 else 0.5
        c_dis_expected = c_retry_overhead * ratio_dis
        c_loss_expected = c_retry_overhead * (1 - ratio_dis)
    else:
        c_dis_expected = 0
        c_loss_expected = 0

    cost_items = ['零配件采购+检测', '装配成本\n(半成品+成品)', '半成品/成品\n检测成本',
                  '拆解费用\n(期望)', '调换损失\n(期望)']
    cost_vals = [c_comps, c_assy_total, c_test_semi_final, c_dis_expected, c_loss_expected]
    total = sum(cost_vals)
    cost_pct = [v / total * 100 for v in cost_vals]

    colors_cost = COLORS[:5]
    wedges, texts, autotexts = ax.pie(cost_vals, labels=cost_items, colors=colors_cost,
                                        autopct='%1.1f%%', textprops={'fontsize': 10})
    ax.set_title(f'总期望成本: {final.effective_cost:.1f} 元/件 | 利润: {profit:.1f} 元/件', fontsize=12)
    savefig(fig, 'problem3_cost_breakdown.png')
    plt.close(fig)

    return comp_dec, semi_dec, final_dec, profit


if __name__ == '__main__':
    solve_problem3()
