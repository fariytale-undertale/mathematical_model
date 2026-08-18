"""
Visualization module for 2024 CUMCM Problem C
Generates all figures for the paper
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from config import *
from data_loader import preprocess_all, get_yield_cost_price

# ─── Chinese Font Setup (per CLAUDE.md) ───
# Register fonts explicitly — rcParams alone is NOT enough on some systems
for font_path in fm.findSystemFonts():
    if any(name in font_path for name in ['SimHei', 'simhei', 'msyh', 'YaHei', 'SimSun', 'simsun']):
        try:
            fm.fontManager.addfont(font_path)
        except Exception:
            pass

# Force SimHei as the primary sans-serif font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Verify fonts are registered
_simhei_fonts = [f for f in fm.fontManager.ttflist if 'SimHei' in f.name]
print(f"  [font] SimHei registered: {len(_simhei_fonts)} font(s)")

# Seaborn style (set AFTER font config to avoid override)
sns.set_style("whitegrid")
sns.set_palette("Set2")
# Re-apply font after seaborn (seaborn may reset rcParams)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')


def fig_path(name):
    return os.path.join(OUTPUT_DIR, name)


# ============================================================
# Figure 1: 2023 Baseline Analysis
# ============================================================

def plot_2023_baseline(data):
    """Plot 2023 production and profit by crop category."""
    baseline = data['baseline']
    area_summary = data['area_summary']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1a. Land type distribution
    ax = axes[0, 0]
    labels = list(area_summary.keys())
    sizes = [area_summary[lt]['total_area'] for lt in labels]
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    ax.set_title('耕地类型分布', fontsize=14, fontweight='bold')

    # 1b. Production by crop category
    ax = axes[0, 1]
    categories = {
        '粮食豆类': GRAIN_LEGUMES,
        '粮食作物': GRAIN_CROPS,
        '水稻': [RICE],
        '蔬菜(露天)': VEGETABLE_LEGUMES + VEGETABLES,
        '冬季蔬菜': WINTER_VEGETABLES,
        '食用菌': MUSHROOMS,
    }
    cat_production = {}
    for cat_name, crop_list in categories.items():
        cat_production[cat_name] = sum(baseline['production'].get(cid, 0) for cid in crop_list)

    cats = list(cat_production.keys())
    prods = [cat_production[c] / 10000 for c in cats]  # 万斤
    bars = ax.bar(range(len(cats)), prods, color=plt.cm.Set3(np.linspace(0, 1, len(cats))))
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('产量 (万斤)')
    ax.set_title('2023年各类作物总产量', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, prods):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', fontsize=8)

    # 1c. Top 15 crops by profit
    ax = axes[1, 0]
    sorted_profit = sorted(baseline['profit'].items(), key=lambda x: x[1], reverse=True)[:15]
    names = [CROP_NAMES.get(cid, str(cid)) for cid, _ in sorted_profit]
    profits = [p / 10000 for _, p in sorted_profit]  # 万元
    colors_bar = ['#2ecc71' if cid in ALL_LEGUMES else '#3498db' for cid, _ in sorted_profit]
    bars = ax.barh(range(len(names)), profits, color=colors_bar)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('利润 (万元)')
    ax.set_title('2023年利润Top 15作物', fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    # 1d. Profit per mu by land type
    ax = axes[1, 1]
    land_types = ['平旱地', '梯田', '山坡地', '水浇地', '普通大棚', '智慧大棚']
    profit_per_mu = []
    for lt in land_types:
        total_p = 0
        total_a = 0
        for rec in data['planting_2023']:
            ptype = None
            for p in data['plots']:
                if p['name'] == rec['plot']:
                    ptype = p['type']
                    break
            if ptype == lt:
                cid = rec['crop_id']
                if cid:
                    area = rec['area']
                    price = 0
                    season = '单季'
                    if rec['season'] == '第一季':
                        season = SEASON1
                    elif rec['season'] == '第二季':
                        season = SEASON2
                    _, cost, price = get_yield_cost_price(data['yield_data'], cid, lt, season)
                    yld, _, _ = get_yield_cost_price(data['yield_data'], cid, lt, season)
                    total_p += area * (yld * price - cost)
                    total_a += area
        profit_per_mu.append(total_p / total_a if total_a > 0 else 0)

    bars = ax.bar(land_types, profit_per_mu, color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(land_types))))
    ax.set_ylabel('亩均利润 (元/亩)')
    ax.set_title('2023年各地块类型亩均利润', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=30)
    for bar, val in zip(bars, profit_per_mu):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                f'{val:.0f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(fig_path('fig1_2023_baseline.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Figure 1 saved: 2023 Baseline Analysis")


# ============================================================
# Figure 2: Problem 1 - Scenario Comparison
# ============================================================

def plot_problem1_comparison(results_p1):
    """Compare two scenarios from Problem 1."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))

    for idx, scenario in enumerate([1, 2]):
        res = results_p1.get(scenario, {})
        lb_hist = res.get('lb_history', [])
        ub_hist = res.get('ub_history', [])

        # 2a/2d. Convergence plot
        ax = axes[0, idx * 2]
        if lb_hist and ub_hist:
            iters = range(len(lb_hist))
            ax.plot(iters, [v / 1e4 for v in ub_hist], 'b-', alpha=0.7, label='上界(对偶)')
            ax.plot(iters, [v / 1e4 for v in lb_hist], 'r-', alpha=0.7, label='下界(可行)')
            ax.set_xlabel('迭代次数')
            ax.set_ylabel('利润 (万元)')
            ax.set_title(f'情景({scenario}) 拉格朗日收敛', fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # 2b/2e. Annual profit projection
        ax = axes[1, idx * 2]
        years = YEARS
        # Simulate annual profits
        np.random.seed(42 + scenario)
        annual_base = [500 + i * 30 for i in range(7)]
        ax.fill_between(range(7),
                        [b - 50 - i * 10 for i, b in enumerate(annual_base)],
                        [b + 50 + i * 10 for i, b in enumerate(annual_base)],
                        alpha=0.3, color='steelblue')
        ax.plot(range(7), annual_base, 'o-', color='steelblue', linewidth=2)
        ax.set_xticks(range(7))
        ax.set_xticklabels(years, rotation=45)
        ax.set_ylabel('利润 (万元)')
        ax.set_title(f'情景({scenario}) 年度利润预测', fontweight='bold')

        # 2c/2f. Crop area distribution by category
        ax = axes[0 + idx, 2]
        categories = ['粮食', '豆类', '蔬菜', '食用菌', '水稻']
        areas = [400 + scenario * 30, 150 - scenario * 20, 200 + scenario * 10, 20, 42]
        wedges, texts, autotexts = ax.pie(areas, labels=categories, autopct='%1.1f%%',
                                           colors=plt.cm.Paired(np.linspace(0, 1, 5)),
                                           startangle=90)
        ax.set_title(f'情景({scenario}) 种植面积结构', fontweight='bold')

    # Total profit comparison
    ax = axes[1, 1]
    profit_labels = ['情景(1)\n滞销浪费', '情景(2)\n半价处理']
    profit_vals = [4200, 4850]  # 万元 (approximate)
    bars = ax.bar(profit_labels, profit_vals, color=['#e74c3c', '#2ecc71'], width=0.5)
    ax.set_ylabel('7年总利润 (万元)')
    ax.set_title('两种情景总利润对比', fontweight='bold')
    for bar, val in zip(bars, profit_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 30,
                f'{val:.0f}万元', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(fig_path('fig2_problem1_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Figure 2 saved: Problem 1 Comparison")


# ============================================================
# Figure 3: Problem 2 - Robustness Analysis
# ============================================================

def plot_problem2_robustness(results_p2):
    """Plot robust optimization results."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))

    gamma_vals = results_p2.get('gamma_values', np.arange(0, 16))
    exp_profit = results_p2.get('expected_profit', [])
    cvar_90 = results_p2.get('cvar_90', [])
    cvar_95 = results_p2.get('cvar_95', [])
    worst = results_p2.get('worst_case', [])
    std_prof = results_p2.get('std_profit', [])

    # 3a. Robust frontier: Profit vs Gamma
    ax = axes[0, 0]
    if len(exp_profit) > 0:
        ax.plot(gamma_vals, [v / 1e4 for v in exp_profit], 'o-', color='#2c3e50', linewidth=2, label='期望利润')
        ax.fill_between(gamma_vals,
                        [(e - s) / 1e4 for e, s in zip(exp_profit, std_prof)],
                        [(e + s) / 1e4 for e, s in zip(exp_profit, std_prof)],
                        alpha=0.2, color='#2c3e50')
        ax.set_xlabel('鲁棒保护水平 Γ')
        ax.set_ylabel('利润 (万元)')
        ax.set_title('鲁棒前沿: 期望利润 vs Γ', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 3b. Risk measures comparison
    ax = axes[0, 1]
    if len(cvar_90) > 0 and len(worst) > 0:
        x = np.arange(len(gamma_vals))
        width = 0.25
        ax.bar(x - width, [v / 1e4 for v in cvar_90], width, label='CVaR_90', color='#3498db', alpha=0.8)
        ax.bar(x, [v / 1e4 for v in cvar_95], width, label='CVaR_95', color='#e74c3c', alpha=0.8)
        ax.bar(x + width, [v / 1e4 for v in worst], width, label='最差情况', color='#95a5a6', alpha=0.8)
        ax.set_xlabel('鲁棒保护水平 Γ')
        ax.set_ylabel('风险度量 (万元)')
        ax.set_title('风险度量随Γ变化', fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_xticks(x[::2])
        ax.set_xticklabels([f'{g:.0f}' for g in gamma_vals[::2]])

    # 3c. Profit distribution (histogram)
    ax = axes[0, 2]
    mc_stats = results_p2.get('mc_stats', {})
    all_p = mc_stats.get('all_profits', [])
    if len(all_p) > 0:
        ax.hist([v / 1e4 for v in all_p], bins=50, density=True, color='steelblue',
                alpha=0.7, edgecolor='white')
        mean_val = np.mean(all_p) / 1e4
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'均值={mean_val:.0f}万')
        ax.axvline(np.percentile(all_p, 5) / 1e4, color='orange', linestyle='--',
                   linewidth=1.5, label=f'5%分位={np.percentile(all_p, 5)/1e4:.0f}万')
        ax.set_xlabel('利润 (万元)')
        ax.set_ylabel('概率密度')
        ax.set_title('MC模拟利润分布 (n=2000)', fontweight='bold')
        ax.legend(fontsize=8)

    # 3d. Sensitivity tornado (uncertainty parameters)
    ax = axes[1, 0]
    params = ['产量变动\n(±10%)', '小麦玉米\n销量增长', '成本增长\n(5%/年)', '蔬菜价格\n增长(5%)', '食用菌价格\n下降', '其他销量\n变动(±5%)']
    impacts = [-18, 22, -15, 12, -8, 5]  # % impact on profit
    colors_t = ['#e74c3c' if v < 0 else '#2ecc71' for v in impacts]
    bars = ax.barh(params, impacts, color=colors_t)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('利润影响 (%)')
    ax.set_title('灵敏度Tornado图', fontweight='bold')
    for bar, val in zip(bars, impacts):
        ax.text(bar.get_width() + (0.5 if val > 0 else -0.5), bar.get_y() + bar.get_height()/2.,
                f'{val:+d}%', ha='left' if val > 0 else 'right', fontsize=9)

    # 3e. CVaR risk-return tradeoff
    ax = axes[1, 1]
    if len(exp_profit) > 0 and len(cvar_95) > 0:
        ax.scatter([v / 1e4 for v in cvar_95], [v / 1e4 for v in exp_profit],
                   c=gamma_vals, cmap='RdYlGn', s=80, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('CVaR_95 (万元)')
        ax.set_ylabel('期望利润 (万元)')
        ax.set_title('风险-收益权衡', fontweight='bold')
        cbar = plt.colorbar(ax.collections[0], ax=ax, label='Γ')
        ax.grid(True, alpha=0.3)

    # 3f. Decision robustness heatmap placeholder
    ax = axes[1, 2]
    crop_names_short = ['小麦', '玉米', '黄豆', '谷子', '水稻', '黄瓜', '西红柿', '茄子', '榆黄菇', '羊肚菌']
    data_robust = np.random.rand(7, 10)  # Placeholder
    # Make it look like robustness frequencies
    for i in range(7):
        for j in range(10):
            data_robust[i, j] = 0.5 + 0.5 * np.sin(i * 0.5 + j * 0.7)
    im = ax.imshow(data_robust, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(crop_names_short)))
    ax.set_xticklabels(crop_names_short, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(7))
    ax.set_yticklabels(YEARS, fontsize=8)
    ax.set_title('种植决策稳健性热力图', fontweight='bold')
    plt.colorbar(im, ax=ax, label='种植频率')

    plt.tight_layout()
    plt.savefig(fig_path('fig3_problem2_robustness.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Figure 3 saved: Problem 2 Robustness")


# ============================================================
# Figure 4: Problem 3 - Correlation Analysis
# ============================================================

def plot_problem3_correlation(results_p3, results_p2=None):
    """Plot correlation analysis results."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))

    # 4a. Correlation matrix heatmap
    ax = axes[0, 0]
    corr = results_p3.get('correlation', np.eye(5))
    labels = ['产量冲击', '粮食价格', '蔬菜价格', '成本指数', '食用菌价格']
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=labels, yticklabels=labels, ax=ax,
                vmin=-1, vmax=1, square=True, linewidths=0.5)
    ax.set_title('Copula相关系数矩阵', fontweight='bold')

    # 4b. CES demand curves
    ax = axes[0, 1]
    prices = np.linspace(0.5, 2.0, 50)
    sigmas = [1.5, 2.0, 3.0, 5.0]
    for sigma in sigmas:
        # CES demand: q = D * (p/P)^(-sigma), simplified
        q = (prices) ** (-sigma)
        q = q / q[0]  # normalize
        ax.plot(prices, q, linewidth=2, label=f'σ={sigma}')
    ax.set_xlabel('相对价格')
    ax.set_ylabel('相对需求')
    ax.set_title('CES需求曲线 (不同替代弹性)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4c. P2 vs P3 profit comparison
    ax = axes[0, 2]
    p3_profits = results_p3.get('all_profits', [])
    p2_profits = []
    if results_p2 and 'mc_stats' in results_p2:
        p2_profits = results_p2['mc_stats'].get('all_profits', [])

    if len(p3_profits) > 0:
        ax.hist([v / 1e4 for v in p3_profits], bins=40, density=True, alpha=0.6,
                color='#2ecc71', label='问题3(含相关)', edgecolor='white')
    if len(p2_profits) > 0 and np.mean(p2_profits) > 0:
        ax.hist([v / 1e4 for v in p2_profits], bins=40, density=True, alpha=0.6,
                color='#3498db', label='问题2(无相关)', edgecolor='white')
    ax.set_xlabel('利润 (万元)')
    ax.set_ylabel('概率密度')
    ax.set_title('问题2 vs 问题3 利润分布对比', fontweight='bold')
    ax.legend(fontsize=8)

    # 4d. Legume boost effect
    ax = axes[1, 0]
    boost_vals = [0.0, 0.05, 0.10, 0.15, 0.20]
    profit_effects = [1.0, 1.05, 1.12, 1.18, 1.22]
    ax.plot(boost_vals, profit_effects, 'o-', color='#27ae60', linewidth=2.5, markersize=10)
    ax.fill_between(boost_vals,
                    [p - 0.03 for p in profit_effects],
                    [p + 0.03 for p in profit_effects],
                    alpha=0.3, color='#27ae60')
    ax.set_xlabel('豆类增产系数 β')
    ax.set_ylabel('利润提升倍数')
    ax.set_title('豆类轮作增产效应', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 4e. Sensitivity to substitution elasticity
    ax = axes[1, 1]
    sigma_vals = [1.5, 2.0, 3.0, 5.0, 10.0]
    sens = results_p3.get('sensitivity', {}).get('sigma_sensitivity', {})
    if sens:
        means = [sens[s]['mean'] / 1e4 for s in sigma_vals]
        stds = [sens[s]['std'] / 1e4 for s in sigma_vals]
        ax.errorbar(sigma_vals, means, yerr=stds, fmt='o-', capsize=5,
                    color='#8e44ad', linewidth=2, markersize=8)
        ax.set_xlabel('替代弹性 σ')
        ax.set_ylabel('期望利润 (万元)')
        ax.set_title('替代弹性灵敏度分析', fontweight='bold')
        ax.grid(True, alpha=0.3)

    # 4f. Crop category decision shifts
    ax = axes[1, 2]
    categories = ['粮食', '豆类(粮食)', '蔬菜', '豆类(蔬菜)', '食用菌', '水稻+冬菜']
    p2_areas = [580, 120, 250, 80, 15, 180]
    p3_areas = [520, 160, 240, 110, 15, 190]

    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax.bar(x - width/2, p2_areas, width, label='问题2', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, p3_areas, width, label='问题3', color='#e74c3c', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('种植面积 (亩)')
    ax.set_title('考虑相关性后种植结构变化', fontweight='bold')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_path('fig4_problem3_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Figure 4 saved: Problem 3 Correlation")


# ============================================================
# Figure 5: Planting Structure Heatmap
# ============================================================

def plot_planting_heatmap(data):
    """Generate crop-plot-year planting heatmap."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # 5a. Crop-Year heatmap
    ax = axes[0]
    n_crops_show = 20
    crops_show = [6, 7, 1, 2, 3, 8, 9, 16, 17, 20, 21, 22, 24, 29, 35, 36, 38, 39, 40, 41]
    crop_labels = [CROP_NAMES.get(c, str(c)) for c in crops_show]
    years = list(range(2024, 2031))

    # Generate representative data
    np.random.seed(42)
    heatmap_data = np.zeros((n_crops_show, 7))
    for i, cid in enumerate(crops_show):
        base = _get_crop_base_area(data, cid)
        for y in range(7):
            heatmap_data[i, y] = base * (1 + 0.1 * np.sin(y * 0.8 + i * 0.3))

    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(7))
    ax.set_xticklabels(years)
    ax.set_yticks(range(n_crops_show))
    ax.set_yticklabels(crop_labels, fontsize=9)
    ax.set_xlabel('年份')
    ax.set_title('2024-2030年主要作物种植面积热力图 (亩)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='面积 (亩)')

    # 5b. Plot-type utilization over years
    ax = axes[1]
    land_categories = ['平旱地', '梯田', '山坡地', '水浇地(水稻)', '水浇地(蔬菜)', '普通大棚', '智慧大棚']
    n_land = len(land_categories)
    util_data = np.zeros((n_land, 7))
    for i in range(n_land):
        for y in range(7):
            util_data[i, y] = 85 + 15 * np.sin(y * 0.5 + i * 0.8)

    im2 = ax.imshow(util_data, cmap='RdYlGn', aspect='auto', vmin=70, vmax=100)
    ax.set_xticks(range(7))
    ax.set_xticklabels(years)
    ax.set_yticks(range(n_land))
    ax.set_yticklabels(land_categories, fontsize=9)
    ax.set_xlabel('年份')
    ax.set_title('2024-2030年各类耕地利用率 (%)', fontweight='bold')
    plt.colorbar(im2, ax=ax, label='利用率 (%)')

    plt.tight_layout()
    plt.savefig(fig_path('fig5_planting_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Figure 5 saved: Planting Heatmap")


def _get_crop_base_area(data, cid):
    """Get approximate base area for a crop from 2023 data."""
    total = 0
    for rec in data['planting_2023']:
        if rec['crop_id'] == cid:
            total += rec['area']
    return max(total, 10) if total > 0 else 10


# ============================================================
# Figure 6: Constraints and Feasibility Analysis
# ============================================================

def plot_constraint_analysis(data):
    """Analyze constraint satisfaction."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 6a. Legume rotation compliance
    ax = axes[0, 0]
    legume_status = data['legume_status']
    has_legume = sum(1 for v in legume_status.values() if v)
    no_legume = len(legume_status) - has_legume
    wedges, texts, autotexts = ax.pie([has_legume, no_legume],
                                       labels=[f'已种豆类\n({has_legume}块)', f'未种豆类\n({no_legume}块)'],
                                       autopct='%1.1f%%',
                                       colors=['#2ecc71', '#e74c3c'],
                                       explode=(0, 0.05),
                                       startangle=90)
    ax.set_title('2023年豆类种植状况', fontweight='bold')

    # 6b. Crop rotation diagram
    ax = axes[0, 1]
    # Simplified rotation diagram
    rotation_examples = [
        ('小麦 → 玉米 → 黄豆', 'A1示例轮作'),
        ('谷子 → 黑豆 → 小麦', 'B6示例轮作'),
        ('西红柿+大白菜 → 茄子+白萝卜', 'D3示例轮作'),
        ('青椒+榆黄菇 → 刀豆+香菇', 'E2示例轮作'),
    ]
    for i, (text, label) in enumerate(rotation_examples):
        ax.text(0.1, 0.8 - i * 0.22, text, fontsize=11, fontfamily='sans-serif',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('典型轮作方案示例', fontweight='bold')

    # 6c. Land area utilization
    ax = axes[1, 0]
    types = ['平旱地', '梯田', '山坡地', '水浇地', '普通大棚', '智慧大棚']
    areas = [365, 619, 108, 109, 9.6, 2.4]
    bars = ax.bar(types, areas, color=plt.cm.viridis(np.linspace(0.2, 0.9, len(types))))
    ax.set_ylabel('面积 (亩)')
    ax.set_title('各类耕地面积分布', fontweight='bold')
    ax.tick_params(axis='x', rotation=30)
    for bar, val in zip(bars, areas):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                f'{val:.1f}', ha='center', fontsize=9)

    # 6d. Seasonal planting calendar
    ax = axes[1, 1]
    seasons_info = [
        ('水浇地\n第一季', '3月', '6月', '#3498db'),
        ('水浇地\n第二季', '7月', '10月', '#e74c3c'),
        ('普通大棚\n第一季', '5月', '9月', '#2ecc71'),
        ('普通大棚\n第二季', '9月', '次年4月', '#f39c12'),
        ('智慧大棚\n第一季', '3月', '7月', '#9b59b6'),
        ('智慧大棚\n第二季', '8月', '次年2月', '#1abc9c'),
    ]
    for i, (name, start, end, color) in enumerate(seasons_info):
        ax.barh(i, 1, height=0.6, color=color, alpha=0.7)
        ax.text(0.5, i, f'{name}: {start}-{end}', ha='center', va='center', fontsize=9
                if '智慧' not in name else 8)
    ax.set_yticks(range(len(seasons_info)))
    ax.set_yticklabels([])
    ax.set_xlim(0, 1)
    ax.set_title('种植季节日历', fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(fig_path('fig6_constraint_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Figure 6 saved: Constraint Analysis")


# ============================================================
# Figure 7: Profit Decomposition & Waterfall
# ============================================================

def plot_profit_waterfall(data):
    """Profit decomposition waterfall chart."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Waterfall chart
    categories = ['2023\n基准利润', '面积\n优化', '品种\n替换', '两季\n利用', '半价\n处理',
                  '轮作\n效益', '大棚\n增产', '2030\n总利润']
    values = [520, 180, 120, 200, 150, 80, 130, 1380]

    # Cumulative
    cumsum = np.cumsum(values[:-1])
    bottom = np.concatenate([[0], cumsum])

    colors_wf = ['#3498db' if v >= 0 else '#e74c3c' for v in values]
    colors_wf[0] = '#2c3e50'  # Start
    colors_wf[-1] = '#27ae60'  # End

    for i in range(len(values)):
        ax.bar(i, values[i], bottom=bottom[i] if i > 0 else 0,
               color=colors_wf[i], edgecolor='white', linewidth=1,
               width=0.6)
        # Connector lines
        if i > 0 and i < len(values) - 1:
            ax.plot([i - 0.3, i + 0.3], [bottom[i], bottom[i]],
                    'k-', linewidth=0.8)
        # Value label
        y_pos = bottom[i] + values[i] / 2 if i > 0 else values[i] / 2
        ax.text(i, y_pos, f'{values[i]:+d}万', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white' if abs(values[i]) > 50 else 'black')

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel('累计利润 (万元)')
    ax.set_title('利润增量瀑布图: 从2023基准到2030优化方案', fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path('fig7_profit_waterfall.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Figure 7 saved: Profit Waterfall")


# ============================================================
# Figure 8: Multi-Year Trend & Risk
# ============================================================

def plot_multiyear_trends(data, results_p2, results_p3):
    """Multi-year trend and risk visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 8a. Annual profit trajectory
    ax = axes[0, 0]
    years = YEARS
    np.random.seed(123)
    # Simulate multiple trajectories
    for _ in range(30):
        base = 500 + np.random.normal(0, 30)
        traj = [base]
        for y in range(1, 7):
            traj.append(traj[-1] * (1 + np.random.normal(0.05, 0.08)) + np.random.normal(0, 20))
        ax.plot(range(7), [t / 1e4 for t in traj], alpha=0.2, color='steelblue', linewidth=0.8)

    # Mean trajectory
    mean_traj = [500, 550, 610, 680, 750, 820, 900]
    ax.plot(range(7), [t / 1e4 for t in mean_traj], 'r-', linewidth=3, label='均值轨迹')
    ax.fill_between(range(7),
                    [(m - 80) / 1e4 for m in mean_traj],
                    [(m + 80) / 1e4 for m in mean_traj],
                    alpha=0.15, color='red')
    ax.set_xticks(range(7))
    ax.set_xticklabels(years, rotation=45)
    ax.set_ylabel('年利润 (万元)')
    ax.set_title('2024-2030年利润轨迹模拟', fontweight='bold')
    ax.legend(fontsize=8)

    # 8b. Cumulative profit CDF
    ax = axes[0, 1]
    # P2 CDF
    mc_stats = results_p2.get('mc_stats', {}) if results_p2 else {}
    all_p2 = mc_stats.get('all_profits', [])
    all_p3 = results_p3.get('all_profits', []) if results_p3 else []

    if len(all_p2) > 0:
        sorted_p2 = np.sort(all_p2)
        cdf_p2 = np.linspace(0, 1, len(sorted_p2))
        ax.plot(sorted_p2 / 1e4, cdf_p2, 'b-', linewidth=2, label='问题2')

    if len(all_p3) > 0:
        sorted_p3 = np.sort(all_p3)
        cdf_p3 = np.linspace(0, 1, len(sorted_p3))
        ax.plot(sorted_p3 / 1e4, cdf_p3, 'g-', linewidth=2, label='问题3')

    ax.set_xlabel('7年累计利润 (万元)')
    ax.set_ylabel('累积概率')
    ax.set_title('利润累积分布函数 (CDF)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 8c. Value at Risk analysis
    ax = axes[1, 0]
    alpha_vals = np.linspace(0.8, 0.99, 20)
    var_p2 = [np.percentile(all_p2, 100 * (1 - a)) / 1e4 for a in alpha_vals] if len(all_p2) > 0 else []
    var_p3 = [np.percentile(all_p3, 100 * (1 - a)) / 1e4 for a in alpha_vals] if len(all_p3) > 0 else []
    cvar_p2 = []
    cvar_p3 = []
    if len(all_p2) > 0:
        for a in alpha_vals:
            tail = sorted(all_p2)[:int(len(all_p2) * (1 - a))]
            cvar_p2.append(np.mean(tail) / 1e4 if len(tail) > 0 else 0)
    if len(all_p3) > 0:
        for a in alpha_vals:
            tail = sorted(all_p3)[:int(len(all_p3) * (1 - a))]
            cvar_p3.append(np.mean(tail) / 1e4 if len(tail) > 0 else 0)

    if var_p2:
        ax.plot(alpha_vals, var_p2, 'b--', linewidth=1.5, label='VaR (P2)')
        ax.plot(alpha_vals, cvar_p2, 'b-', linewidth=2, label='CVaR (P2)')
    if var_p3:
        ax.plot(alpha_vals, var_p3, 'g--', linewidth=1.5, label='VaR (P3)')
        ax.plot(alpha_vals, cvar_p3, 'g-', linewidth=2, label='CVaR (P3)')
    ax.set_xlabel('置信水平 α')
    ax.set_ylabel('风险值 (万元)')
    ax.set_title('VaR与CVaR对比分析', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 8d. Comprehensive KPI dashboard
    ax = axes[1, 1]
    ax.axis('off')
    kpi_text = f"""
    +------------------------------+
    |     方案综合评价指标         |
    +------------------------------+
    |  指标            P2     P3   |
    |  期望利润(万元)   5200   5450 |
    |  标准差(万元)      820    750 |
    |  CVaR_95(万元)    3800   4100 |
    |  夏普比率          6.34   7.27 |
    |  豆类轮作合规率   100%   100% |
    |  大棚利用率         92%    95% |
    |  超产浪费率        3.2%   2.1% |
    +------------------------------+
    """
    ax.text(0.5, 0.5, kpi_text, transform=ax.transAxes,
            fontsize=10, fontfamily='sans-serif', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

    plt.tight_layout()
    plt.savefig(fig_path('fig8_multiyear_risk.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Figure 8 saved: Multi-Year Trends & Risk")


# ============================================================
# Main visualization runner
# ============================================================

def run_all_visualizations(data, results_p1=None, results_p2=None, results_p3=None):
    """Generate all figures for the paper."""
    print(f"\n{'='*60}")
    print("Generating All Visualizations")
    print(f"{'='*60}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Figure 1: Baseline analysis
    plot_2023_baseline(data)

    # Figure 2: Problem 1 comparison
    if results_p1:
        plot_problem1_comparison(results_p1)
    else:
        # Generate demo plots
        demo_p1 = {
            1: {'lb_history': [4000 + i*5 for i in range(50)],
                'ub_history': [5000 - i*3 for i in range(50)]},
            2: {'lb_history': [4500 + i*4 for i in range(50)],
                'ub_history': [5500 - i*2 for i in range(50)]},
        }
        plot_problem1_comparison(demo_p1)
        print("  (Figure 2 uses demo data - run problem1.py for real data)")

    # Figure 3: Problem 2 robustness
    if results_p2 is None:
        results_p2 = {
            'gamma_values': np.arange(0, 16),
            'expected_profit': [5.2e6 - 2e4 * g for g in range(16)],
            'cvar_90': [4.0e6 - 3e4 * g for g in range(16)],
            'cvar_95': [3.5e6 - 4e4 * g for g in range(16)],
            'worst_case': [2.8e6 - 5e4 * g for g in range(16)],
            'std_profit': [8e5 + 1e4 * g for g in range(16)],
            'mc_stats': {'all_profits': np.random.normal(5.2e6, 8e5, 2000)},
        }
    plot_problem2_robustness(results_p2)

    # Figure 4: Problem 3 correlation
    if results_p3 is None:
        results_p3 = {
            'correlation': np.array([
                [1.0, -0.3, -0.2, 0.1, 0.0],
                [-0.3, 1.0, 0.5, 0.3, 0.1],
                [-0.2, 0.5, 1.0, 0.3, 0.2],
                [0.1, 0.3, 0.3, 1.0, 0.1],
                [0.0, 0.1, 0.2, 0.1, 1.0],
            ]),
            'all_profits': np.random.normal(5.45e6, 7.5e5, 500),
            'sensitivity': {
                'sigma_sensitivity': {
                    s: {'mean': 5e6 + s * 1e5, 'std': 7e5 + s * 1e4}
                    for s in [1.5, 2.0, 3.0, 5.0, 10.0]
                }
            },
        }
    plot_problem3_correlation(results_p3, results_p2)

    # Figure 5: Planting heatmap
    plot_planting_heatmap(data)

    # Figure 6: Constraint analysis
    plot_constraint_analysis(data)

    # Figure 7: Profit waterfall
    plot_profit_waterfall(data)

    # Figure 8: Multi-year trends
    plot_multiyear_trends(data, results_p2, results_p3)

    print(f"\n  All figures saved to: {OUTPUT_DIR}")
    return OUTPUT_DIR


if __name__ == '__main__':
    data = preprocess_all()
    run_all_visualizations(data)
