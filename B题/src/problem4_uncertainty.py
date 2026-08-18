"""
Problem 4: Uncertainty Integration
====================================
Assume defect rates from Problems 2 & 3 are estimated via Problem 1's SPRT.
Redo Problems 2 & 3 accounting for sampling uncertainty.

Methods:
1. Bayesian posterior from SPRT sampling → Beta distribution for each p
2. Monte Carlo: sample from posterior, re-solve P2/P3 for each sample
3. DRO with Wasserstein ball: robust optimization against worst-case
4. Sensitivity: report distribution of optimal decisions and profits
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from collections import Counter
from itertools import product
import pandas as pd

from src.config import (
    TABLE1_SCENARIOS, N_MC_SAMPLES, N_SPRT_SAMPLES_HISTORICAL,
    CONFIDENCE_LEVEL, P0_NOMINAL, P1_ALTERNATIVE
)
from src.plot_utils import savefig, new_figure, COLORS, OUTPUT_DIR
from src.problem2_decision import expected_profit, solve_scenario
from src.problem3_tree import build_assembly_tree, evaluate_full_tree


# ============================================================
# Part A: Bayesian Posterior from SPRT
# ============================================================

def sprt_posterior(true_p, n_samples, prior_alpha=1.0, prior_beta=9.0, seed=None):
    """
    Simulate SPRT sampling and return Beta posterior.

    In practice, the SPRT stops when a decision is reached.
    Here we simulate n_samples draws from Bernoulli(true_p) and
    return the Beta posterior.

    Returns
    -------
    alpha_post, beta_post : float
    samples : array of 0/1
    """
    rng = np.random.RandomState(seed)
    samples = (rng.random(n_samples) < true_p).astype(int)
    k = samples.sum()
    alpha_post = prior_alpha + k
    beta_post = prior_beta + (n_samples - k)
    return alpha_post, beta_post, samples


def posterior_to_ci(alpha, beta, confidence=0.95):
    """Compute confidence interval from Beta posterior."""
    lower = stats.beta.ppf((1 - confidence) / 2, alpha, beta)
    upper = stats.beta.ppf(1 - (1 - confidence) / 2, alpha, beta)
    mean = alpha / (alpha + beta)
    return lower, mean, upper


# ============================================================
# Part B: Monte Carlo for Problem 2
# ============================================================

def mc_problem2(n_samples=N_MC_SAMPLES, seed=42):
    """
    Monte Carlo: sample defect rates from Beta posteriors,
    re-optimize Problem 2 for each sample.
    """
    rng = np.random.RandomState(seed)
    print("=" * 60)
    print("问题4: 蒙特卡洛模拟 — 问题2的不确定性分析")
    print("=" * 60)
    print(f"  模拟次数: {n_samples}")
    print(f"  每参数历史抽样量: {N_SPRT_SAMPLES_HISTORICAL}")

    # Store results for each scenario
    all_mc_results = {}

    for s_idx, scenario in enumerate(TABLE1_SCENARIOS):
        print(f"\n--- 情形 {s_idx + 1} ---")

        # True defect rates (unknown to us, but we sample from them)
        true_p1 = scenario['p1']
        true_p2 = scenario['p2']
        true_pf = scenario['pf']

        # Prior: Beta(1, 9) → mean = 0.10 (nominal)
        # Sample from the true distribution to simulate SPRT
        alpha1, beta1, _ = sprt_posterior(true_p1, N_SPRT_SAMPLES_HISTORICAL, seed=seed + s_idx * 1000)
        alpha2, beta2, _ = sprt_posterior(true_p2, N_SPRT_SAMPLES_HISTORICAL, seed=seed + s_idx * 1000 + 100)
        alphaf, betaf, _ = sprt_posterior(true_pf, N_SPRT_SAMPLES_HISTORICAL, seed=seed + s_idx * 1000 + 200)

        # Posterior estimates and CIs
        ci1_l, est1, ci1_u = posterior_to_ci(alpha1, beta1, CONFIDENCE_LEVEL)
        ci2_l, est2, ci2_u = posterior_to_ci(alpha2, beta2, CONFIDENCE_LEVEL)
        cif_l, estf, cif_u = posterior_to_ci(alphaf, betaf, CONFIDENCE_LEVEL)

        print(f"  零配件1: 真实p₁={true_p1}, 估计={est1:.4f}, "
              f"{CONFIDENCE_LEVEL:.0%}CI=[{ci1_l:.4f}, {ci1_u:.4f}]")
        print(f"  零配件2: 真实p₂={true_p2}, 估计={est2:.4f}, "
              f"{CONFIDENCE_LEVEL:.0%}CI=[{ci2_l:.4f}, {ci2_u:.4f}]")
        print(f"  成品:    真实pf={true_pf}, 估计={estf:.4f}, "
              f"{CONFIDENCE_LEVEL:.0%}CI=[{cif_l:.4f}, {cif_u:.4f}]")

        # Monte Carlo: sample from posteriors, re-optimize
        mc_p1 = rng.beta(alpha1, beta1, n_samples)
        mc_p2 = rng.beta(alpha2, beta2, n_samples)
        mc_pf = rng.beta(alphaf, betaf, n_samples)

        decision_counts = Counter()
        profits_mc = []
        decisions_list = []

        for i in range(n_samples):
            sc = dict(scenario)
            sc['p1'] = mc_p1[i]
            sc['p2'] = mc_p2[i]
            sc['pf'] = mc_pf[i]

            best_dec, best_profit, baseline, _ = solve_scenario(sc, s_idx + 1)
            decision_counts[best_dec] += 1
            profits_mc.append(best_profit)
            decisions_list.append(best_dec)

        # Most frequent decision
        top_decision = decision_counts.most_common(1)[0]
        top_pct = top_decision[1] / n_samples * 100

        # Profit statistics
        profits_arr = np.array(profits_mc)
        profit_mean = profits_arr.mean()
        profit_std = profits_arr.std()
        profit_ci = stats.norm.interval(CONFIDENCE_LEVEL, loc=profit_mean, scale=profit_std / np.sqrt(n_samples))

        print(f"  最优决策分布: {dict(decision_counts)}")
        print(f"  最频繁决策: {top_decision[0]} ({top_pct:.1f}%)")
        print(f"  期望利润: {profit_mean:.2f} ± {profit_std:.2f} 元/件")
        print(f"  利润{CONFIDENCE_LEVEL:.0%}CI: [{profit_ci[0]:.2f}, {profit_ci[1]:.2f}]")

        # Also compute the "certainty-equivalent" optimal decision
        # (using point estimates)
        sc_point = dict(scenario)
        sc_point['p1'] = est1
        sc_point['p2'] = est2
        sc_point['pf'] = estf
        point_dec, point_profit, _, _ = solve_scenario(sc_point, s_idx + 1)

        all_mc_results[s_idx] = {
            'scenario': s_idx + 1,
            'true_p': (true_p1, true_p2, true_pf),
            'estimates': (est1, est2, estf),
            'ci_p1': (ci1_l, ci1_u),
            'ci_p2': (ci2_l, ci2_u),
            'ci_pf': (cif_l, cif_u),
            'decision_counts': decision_counts,
            'top_decision': top_decision,
            'top_pct': top_pct,
            'profit_mean': profit_mean,
            'profit_std': profit_std,
            'profit_ci': profit_ci,
            'point_decision': point_dec,
            'point_profit': point_profit,
            'profits_mc': profits_arr,
            'decisions_list': decisions_list,
        }

    # --- Visualization ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('问题4: 问题2各情形期望利润的蒙特卡洛分布', fontsize=14, fontweight='bold')

    for s_idx in range(6):
        ax = axes[s_idx // 3, s_idx % 3]
        res = all_mc_results[s_idx]
        profits = res['profits_mc']

        ax.hist(profits, bins=40, density=True, alpha=0.7, color=COLORS[s_idx % len(COLORS)],
                edgecolor='white', linewidth=0.5)
        ax.axvline(res['profit_mean'], color='red', linestyle='--', linewidth=2,
                   label=f"均值={res['profit_mean']:.1f}")
        ax.axvline(res['point_profit'], color='green', linestyle=':', linewidth=2,
                   label=f"点估计={res['point_profit']:.1f}")

        # Also show true-profit (with known params)
        true_profit = expected_profit(TABLE1_SCENARIOS[s_idx], *res['top_decision'][0])
        ax.axvline(true_profit, color='blue', linestyle='-', linewidth=1.5, alpha=0.5,
                   label=f"真值={true_profit:.1f}")

        ax.set_title(f"情形 {s_idx + 1}")
        ax.set_xlabel('期望利润 (元/件)')
        ax.set_ylabel('概率密度')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    savefig(fig, 'problem4_mc_profit_distributions.png')
    plt.close(fig)

    # --- Decision robustness bar chart ---
    fig, ax = new_figure((14, 6), title='问题4: 各情形最优决策的蒙特卡洛频率分布')
    decision_labels = []
    decision_freqs = []
    for s_idx in range(6):
        counts = all_mc_results[s_idx]['decision_counts']
        total = sum(counts.values())
        for dec, count in counts.most_common(5):
            decision_labels.append(f"S{s_idx+1}:{dec}")
            decision_freqs.append(count / total * 100)

    # Plot top decisions
    n_show = min(30, len(decision_labels))
    colors = [COLORS[i // 5 % len(COLORS)] for i in range(n_show)]
    ax.barh(range(n_show), decision_freqs[:n_show], color=colors, alpha=0.8, edgecolor='white')
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(decision_labels[:n_show], fontsize=8)
    ax.set_xlabel('频率 (%)')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2, axis='x')
    savefig(fig, 'problem4_decision_frequencies.png')
    plt.close(fig)

    return all_mc_results


# ============================================================
# Part C: Monte Carlo for Problem 3
# ============================================================

def mc_problem3(n_samples=500, seed=123):
    """Monte Carlo for Problem 3 assembly tree with uncertain defect rates."""
    rng = np.random.RandomState(seed)
    print("\n" + "=" * 60)
    print("问题4: 蒙特卡洛模拟 — 问题3的不确定性分析")
    print("=" * 60)
    print(f"  模拟次数: {n_samples}")

    # True defect rates from Table 2
    true_ps_comp = [0.10] * 8  # all components have p=0.10
    true_ps_semi = [0.10, 0.10, 0.10]
    true_pf = 0.10

    # Generate posteriors
    posteriors = []
    for i, p in enumerate(true_ps_comp):
        a, b, _ = sprt_posterior(p, N_SPRT_SAMPLES_HISTORICAL, seed=seed + i)
        posteriors.append((a, b))

    semi_posteriors = []
    for i, p in enumerate(true_ps_semi):
        a, b, _ = sprt_posterior(p, N_SPRT_SAMPLES_HISTORICAL, seed=seed + 100 + i)
        semi_posteriors.append((a, b))

    a_f, b_f, _ = sprt_posterior(true_pf, N_SPRT_SAMPLES_HISTORICAL, seed=seed + 200)
    final_posterior = (a_f, b_f)

    # MC sampling
    profits_mc3 = []
    comp_decision_counts = [Counter() for _ in range(8)]
    semi_decision_counts = [Counter() for _ in range(3)]

    for i in range(n_samples):
        # Sample from posteriors
        p_comps = [rng.beta(a, b) for a, b in posteriors]
        p_semis = [rng.beta(a, b) for a, b in semi_posteriors]
        p_final = rng.beta(a_f, b_f)

        # Build modified tree
        components, semi_products, final = build_assembly_tree()

        # Override defect rates
        for j, comp in enumerate(components):
            comp.p_defect = p_comps[j]
        for j, semi in enumerate(semi_products):
            semi.p_defect = p_semis[j]
        final.p_defect = p_final

        # Solve using full enumeration over component testing decisions
        n_comp = len(components)
        best_profit = -np.inf
        best_comp_test = None
        for test_mask in range(2**n_comp):
            comp_test_vec = [(test_mask >> j) & 1 for j in range(n_comp)]
            profit, _, _, _ = evaluate_full_tree(
                comp_test_vec, components, semi_products, final)
            if profit > best_profit:
                best_profit = profit
                best_comp_test = comp_test_vec

        profits_mc3.append(best_profit)

        # Record decisions
        for j in range(8):
            comp_decision_counts[j][bool(best_comp_test[j])] += 1

    # Results
    profits_arr = np.array(profits_mc3)
    print(f"\n  期望利润: {profits_arr.mean():.2f} ± {profits_arr.std():.2f} 元/件")
    print(f"  {CONFIDENCE_LEVEL:.0%}CI: [{np.percentile(profits_arr, 2.5):.2f}, "
          f"{np.percentile(profits_arr, 97.5):.2f}]")

    print("\n  零配件检测决策频率:")
    for j in range(8):
        counts = comp_decision_counts[j]
        total = sum(counts.values())
        test_pct = counts.get(True, 0) / total * 100
        print(f"    零配件{j+1}: 检测={test_pct:.1f}%, 不检测={100-test_pct:.1f}%")

    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Profit distribution
    ax = axes[0]
    ax.hist(profits_arr, bins=40, density=True, alpha=0.7, color=COLORS[0], edgecolor='white')
    ax.axvline(profits_arr.mean(), color='red', linestyle='--', linewidth=2,
               label=f"均值={profits_arr.mean():.1f}")
    ax.set_xlabel('期望利润 (元/件)')
    ax.set_ylabel('概率密度')
    ax.set_title('问题3 成品期望利润分布')
    ax.legend()
    ax.grid(True, alpha=0.2)

    # Component test decision robustness
    ax = axes[1]
    test_rates = []
    for j in range(8):
        counts = comp_decision_counts[j]
        total = sum(counts.values())
        test_rates.append(counts.get(True, 0) / total * 100)

    colors_bar = [COLORS[0] if r > 50 else COLORS[1] for r in test_rates]
    ax.bar(range(8), test_rates, color=colors_bar, alpha=0.8, edgecolor='white')
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(range(8))
    ax.set_xticklabels([f'C{i+1}' for i in range(8)])
    ax.set_ylabel('检测决策频率 (%)')
    ax.set_title('零配件检测决策鲁棒性')
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    savefig(fig, 'problem4_mc_problem3.png')
    plt.close(fig)

    return profits_mc3, comp_decision_counts


def plot_posterior_and_boundary(mc_p2_results):
    """Additional Problem 4 visualizations: posteriors and decision boundaries."""
    from scipy import stats as scipy_stats
    from src.config import N_SPRT_SAMPLES_HISTORICAL

    # --- Posterior distributions for Scenario 1 parameters ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('问题4: 情形1次品率的贝叶斯后验分布', fontsize=14, fontweight='bold')
    sc1 = TABLE1_SCENARIOS[0]
    true_ps = [sc1['p1'], sc1['p2'], sc1['pf']]
    titles = ['零配件1次品率 p1', '零配件2次品率 p2', '成品次品率 pf']
    for j, (tp, title) in enumerate(zip(true_ps, titles)):
        ax = axes[j]
        a_post, b_post, _ = sprt_posterior(tp, N_SPRT_SAMPLES_HISTORICAL, seed=42 + j*100)
        x_grid = np.linspace(0.01, 0.35, 200)
        prior_pdf = scipy_stats.beta.pdf(x_grid, 1, 9)
        post_pdf = scipy_stats.beta.pdf(x_grid, a_post, b_post)
        ax.plot(x_grid, prior_pdf, 'gray', linestyle='--', linewidth=1.5, alpha=0.7, label='先验 Beta(1,9)')
        ax.plot(x_grid, post_pdf, COLORS[0], linewidth=2, label=f'后验 Beta({a_post:.0f},{b_post:.0f})')
        ax.axvline(tp, color='red', linestyle='-', linewidth=1.5, alpha=0.7, label=f'真实值={tp}')
        ax.axvline(a_post/(a_post+b_post), color=COLORS[0], linestyle=':', linewidth=1.5, label=f'后验均值={a_post/(a_post+b_post):.3f}')
        ax.set_title(title); ax.set_xlabel('次品率'); ax.set_ylabel('概率密度')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    savefig(fig, 'problem4_posterior_distributions.png')
    plt.close(fig)

    # --- Decision boundary: p1 vs pf for Scenario 1 ---
    fig, ax = new_figure((9, 7), title='问题4: 参数空间中的最优决策区域 (情形1)')
    p1_g = np.linspace(0.04, 0.25, 50)
    pf_g = np.linspace(0.04, 0.25, 50)
    P1M, PFM = np.meshgrid(p1_g, pf_g)
    Z_dec = np.zeros_like(P1M, dtype=int)
    dec_map = {}
    for i in range(len(pf_g)):
        for j in range(len(p1_g)):
            sc = dict(sc1); sc['p1'] = P1M[i,j]; sc['pf'] = PFM[i,j]
            best_p = -np.inf; best_d = 0
            for idx_d, (xx1, xx2, yy, zz) in enumerate(product([0,1], repeat=4)):
                p = expected_profit(sc, xx1, xx2, yy, zz)
                if p > best_p: best_p = p; best_d = idx_d
            Z_dec[i,j] = best_d
            dec_map[best_d] = (xx1, xx2, yy, zz) if best_d in dec_map else dec_map.get(best_d, None)
    # Re-map to continuous index
    unique_decs = sorted(set(Z_dec.flatten()))
    Z_remap = np.zeros_like(Z_dec, dtype=int)
    for new_idx, old_idx in enumerate(unique_decs):
        Z_remap[Z_dec == old_idx] = new_idx
    cs = ax.contourf(P1M, PFM, Z_remap, levels=np.arange(-0.5, len(unique_decs)+0.5, 1),
                      cmap='Set3', alpha=0.85)
    ax.scatter([sc1['p1']], [sc1['pf']], marker='*', s=300, c='red', edgecolors='black',
               linewidths=1.5, zorder=5, label=f'名义参数 (p1={sc1["p1"]}, pf={sc1["pf"]})')
    ax.set_xlabel('零配件1次品率 p1'); ax.set_ylabel('成品次品率 pf')
    ax.legend(fontsize=9)
    savefig(fig, 'problem4_decision_boundary.png')
    plt.close(fig)


# ============================================================
# Part D: Wasserstein DRO
# ============================================================

def wasserstein_dro_problem2(scenario, epsilon=0.03, n_grid=20):
    """
    Wasserstein Distributionally Robust Optimization for Problem 2.

    For each decision combination, compute the worst-case expected profit
    over defect rates within a Wasserstein ball of radius ε around the
    empirical distribution.
    """
    # Empirical (estimated) defect rates
    p1_hat = scenario['p1']
    p2_hat = scenario['p2']
    pf_hat = scenario['pf']

    # Grid of possible defect rates within ε-ball (L1/W1 bound)
    # For Bernoulli(p), W₁ distance between Bernoulli(p) and Bernoulli(p̂)
    # equals |p - p̂|
    p1_range = np.linspace(max(0.01, p1_hat - epsilon), min(0.30, p1_hat + epsilon), n_grid)
    p2_range = np.linspace(max(0.01, p2_hat - epsilon), min(0.30, p2_hat + epsilon), n_grid)
    pf_range = np.linspace(max(0.01, pf_hat - epsilon), min(0.30, pf_hat + epsilon), n_grid)

    # All decision combinations
    decisions = list(__import__('itertools').product([0, 1], repeat=4))

    dro_results = {}
    for dec in decisions:
        worst_profit = np.inf
        worst_params = None
        for p1 in p1_range:
            for p2 in p2_range:
                for pf in pf_range:
                    # Check Wasserstein constraint
                    w1_dist = abs(p1 - p1_hat) + abs(p2 - p2_hat) + abs(pf - pf_hat)
                    if w1_dist <= epsilon * 3:  # sum of individual ε
                        sc = dict(scenario)
                        sc['p1'] = p1; sc['p2'] = p2; sc['pf'] = pf
                        profit = expected_profit(sc, *dec)
                        if profit < worst_profit:
                            worst_profit = profit
                            worst_params = (p1, p2, pf)

        dro_results[dec] = {'worst_profit': worst_profit, 'worst_params': worst_params}

    # Find the decision with best worst-case profit
    best_dro_dec = max(dro_results, key=lambda d: dro_results[d]['worst_profit'])
    return best_dro_dec, dro_results


def dro_sensitivity(scenario, scenario_idx=1, epsilons=None, n_grid=15):
    """
    Sensitivity analysis for Wasserstein DRO radius epsilon.

    Varies epsilon and reports:
    - Optimal DRO decision at each epsilon
    - Worst-case profit at each epsilon
    - Decision switching points
    """
    if epsilons is None:
        epsilons = np.linspace(0.005, 0.15, 25)

    print(f"\n--- Wasserstein DRO 半径敏感性分析 (情形{scenario_idx}) ---")
    print(f"  epsilon 范围: [{epsilons[0]:.3f}, {epsilons[-1]:.3f}], 共{len(epsilons)}个值")

    results = []
    for eps in epsilons:
        best_dec, dro_res = wasserstein_dro_problem2(scenario, epsilon=eps, n_grid=n_grid)
        worst_profit = dro_res[best_dec]['worst_profit']
        worst_params = dro_res[best_dec]['worst_params']
        results.append({
            'epsilon': eps,
            'best_decision': best_dec,
            'worst_profit': worst_profit,
            'worst_params': worst_params,
        })
        print(f"  ε={eps:.3f}: 最优决策={best_dec}, 最坏利润={worst_profit:.2f}")

    # Identify decision switching points
    switches = []
    for i in range(1, len(results)):
        if results[i]['best_decision'] != results[i-1]['best_decision']:
            switches.append((results[i-1]['epsilon'], results[i-1]['best_decision'],
                            results[i]['epsilon'], results[i]['best_decision']))

    print(f"\n  决策切换点: {len(switches)}处")
    for sw in switches:
        print(f"    ε={sw[0]:.3f}({sw[1]}) → ε={sw[2]:.3f}({sw[3]})")

    # --- Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.suptitle(f'Wasserstein DRO 半径敏感性分析 (情形{scenario_idx})', fontsize=14, fontweight='bold')

    eps_vals = [r['epsilon'] for r in results]
    worst_profits = [r['worst_profit'] for r in results]

    # Get nominal profit for reference
    nominal_dec, nominal_profit, _, _ = solve_scenario(scenario, scenario_idx)

    # Left: worst-case profit vs epsilon
    ax1.plot(eps_vals, worst_profits, 'b-o', markersize=4, linewidth=1.5, label='DRO最坏利润')
    ax1.axhline(y=nominal_profit, color='green', linestyle='--', linewidth=1.5,
                label=f'名义利润={nominal_profit:.1f}')
    # Color regions by decision
    unique_decs = sorted(set(str(r['best_decision']) for r in results))
    dec_colors = {d: COLORS[i % len(COLORS)] for i, d in enumerate(unique_decs)}
    for i in range(len(results)):
        ax1.axvspan(eps_vals[i] - (eps_vals[1]-eps_vals[0])/2 if i > 0 else eps_vals[0],
                     eps_vals[i] + (eps_vals[1]-eps_vals[0])/2 if i < len(eps_vals)-1 else eps_vals[-1],
                     alpha=0.1, color=dec_colors[str(results[i]['best_decision'])])
    ax1.set_xlabel('Wasserstein半径 ε')
    ax1.set_ylabel('最坏情况期望利润 (元/件)')
    ax1.set_title('最坏利润 vs ε')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    # Right: decision stability
    ax2 = plt.subplot(1, 2, 2)
    decision_stability = {}
    for r in results:
        dec_str = str(r['best_decision'])
        decision_stability.setdefault(dec_str, []).append(1)
    # Stack plot
    dec_list = list(decision_stability.keys())
    data = np.zeros((len(dec_list), len(eps_vals)))
    for j, dec in enumerate(dec_list):
        data[j] = [1 if str(r['best_decision']) == dec else 0 for r in results]
    bottom = np.zeros(len(eps_vals))
    for j, dec in enumerate(dec_list):
        ax2.fill_between(eps_vals, bottom, bottom + data[j],
                         alpha=0.7, color=dec_colors[dec], label=f'{dec}')
        bottom += data[j]
    ax2.set_xlabel('Wasserstein半径 ε')
    ax2.set_ylabel('决策占比')
    ax2.set_title('不同ε下的最优决策')
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    savefig(fig, 'problem4_dro_sensitivity.png')
    plt.close(fig)

    # Also do for a second scenario (scenario 3) for comparison
    if scenario_idx == 1:
        print(f"\n--- 对情形3进行DRO敏感性分析 (高调换损失场景) ---")
        sc3 = TABLE1_SCENARIOS[2]
        # Quick scan — fewer epsilons for speed
        eps_quick = np.linspace(0.01, 0.12, 12)
        for eps in eps_quick:
            best_dec, dro_res = wasserstein_dro_problem2(sc3, epsilon=eps, n_grid=12)
            w_prof = dro_res[best_dec]['worst_profit']
            print(f"  ε={eps:.3f}: 决策={best_dec}, 最坏利润={w_prof:.2f}")

    return results, switches


def solve_problem4():
    """Main entry for Problem 4."""
    print("=" * 60)
    print("问题4: 考虑抽样不确定性的综合决策")
    print("=" * 60)

    # MC for Problem 2
    mc_p2 = mc_problem2(n_samples=N_MC_SAMPLES)

    # MC for Problem 3
    mc_p3, comp_counts = mc_problem3(n_samples=500)

    # Additional visualizations
    print("\n--- 生成后验分布与决策边界图 ---")
    plot_posterior_and_boundary(mc_p2)

    # DRO for Problem 2 Scenario 1
    print("\n--- Wasserstein DRO 分析 (问题2情形1) ---")
    best_dro, dro_results = wasserstein_dro_problem2(TABLE1_SCENARIOS[0], epsilon=0.03)

    print(f"  DRO最优决策: {best_dro}")
    print(f"  最坏情况利润: {dro_results[best_dro]['worst_profit']:.2f} 元/件")
    print(f"  最坏参数: {dro_results[best_dro]['worst_params']}")

    # Compare with nominal and MC results
    nominal_dec, nominal_profit, _, _ = solve_scenario(TABLE1_SCENARIOS[0], 1)
    print(f"\n  名义最优决策: {nominal_dec}, 利润={nominal_profit:.2f}")
    print(f"  MC最频繁决策: {mc_p2[0]['top_decision'][0]}, 利润均值={mc_p2[0]['profit_mean']:.2f}")

    # DRO epsilon sensitivity analysis
    print("\n" + "=" * 60)
    print("Wasserstein DRO 半径敏感性分析")
    print("=" * 60)
    dro_sens_results, dro_switches = dro_sensitivity(TABLE1_SCENARIOS[0], scenario_idx=1)

    # --- Summary table ---
    print("\n" + "=" * 60)
    print("问题4 综合结果汇总:")
    print("=" * 60)

    summary_rows = []
    for s_idx in range(6):
        r = mc_p2[s_idx]
        nominal_dec, nominal_profit, _, _ = solve_scenario(TABLE1_SCENARIOS[s_idx], s_idx + 1)
        summary_rows.append({
            '情形': s_idx + 1,
            '名义最优决策': str(nominal_dec),
            '名义利润': f'{nominal_profit:.2f}',
            'MC最频繁决策': str(r['top_decision'][0]),
            'MC频率': f'{r["top_pct"]:.1f}%',
            'MC利润均值': f'{r["profit_mean"]:.2f}',
            'MC利润std': f'{r["profit_std"]:.2f}',
            '利润95%CI': f'[{r["profit_ci"][0]:.2f}, {r["profit_ci"][1]:.2f}]',
        })

    df = pd.DataFrame(summary_rows)
    print(df.to_string(index=False))

    return mc_p2, mc_p3, df, dro_sens_results


if __name__ == '__main__':
    solve_problem4()
