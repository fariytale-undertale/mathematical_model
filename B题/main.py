#!/usr/bin/env python
"""
2024 CUMCM Problem B: Production Decision Optimization
======================================================
Main entry point — solves all 4 sub-problems and generates outputs.

Usage: python main.py [--problem N] [--no-plot]
"""

import sys
import os
import argparse
import time
import json

# Ensure src is on path
sys.path.insert(0, os.path.dirname(__file__))

from src.config import *
from src.problem1_sprt import solve_problem1
from src.problem2_decision import solve_problem2
from src.problem3_tree import solve_problem3
from src.problem4_uncertainty import solve_problem4


def save_results(results, filename):
    """Save results dict to JSON (serializable parts only)."""
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        return obj

    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(convert(results), f, ensure_ascii=False, indent=2, default=str)
        print(f"[Results saved] {path}")
    except Exception as e:
        print(f"[Warning] Could not save results: {e}")


def main():
    parser = argparse.ArgumentParser(description='2024 CUMCM B题 求解')
    parser.add_argument('--problem', type=int, choices=[1, 2, 3, 4], default=0,
                        help='Run specific problem only (0=all)')
    parser.add_argument('--no-plot', action='store_true', help='Skip plot generation')
    args = parser.parse_args()

    start_time = time.time()
    print("=" * 70)
    print("  2024 年高教社杯全国大学生数学建模竞赛 B 题")
    print("  生产过程中的决策问题 — 求解程序")
    print("=" * 70)

    all_results = {}

    if args.problem in (0, 1):
        print("\n" + "█" * 60)
        print("  问题 1: 抽样检测方案设计")
        print("█" * 60)
        r1 = solve_problem1()
        all_results['problem1'] = {
            'p0': r1['p0'], 'p1': r1['p1'],
            'alpha': r1['alpha'], 'beta': r1['beta'],
            'log_A': r1['log_A'], 'log_B': r1['log_B'],
            'avg_n_p0': r1['avg_n_p0'], 'avg_n_p1': r1['avg_n_p1'],
            'reject_rate_at_p0': r1['reject_rate_at_p0'],
            'reject_rate_at_p1': r1['reject_rate_at_p1'],
        }

    if args.problem in (0, 2):
        print("\n" + "█" * 60)
        print("  问题 2: 两零配件生产过程决策优化")
        print("█" * 60)
        df_p2, results_p2 = solve_problem2()
        all_results['problem2'] = results_p2

    if args.problem in (0, 3):
        print("\n" + "█" * 60)
        print("  问题 3: m道工序n个零配件 — 装配树决策优化")
        print("█" * 60)
        comp_dec, semi_dec, final_dec, profit_p3 = solve_problem3()
        all_results['problem3'] = {
            'component_decisions': comp_dec,
            'semi_decisions': semi_dec,
            'final_decision': final_dec,
            'expected_profit': profit_p3,
        }

    if args.problem in (0, 4):
        print("\n" + "█" * 60)
        print("  问题 4: 考虑抽样不确定性的综合决策")
        print("█" * 60)
        mc_p2, mc_p3, df_p4 = solve_problem4()
        all_results['problem4'] = {
            'mc_p2_summary': {str(k): {
                'top_decision': str(v['top_decision'][0]),
                'top_pct': float(v['top_pct']),
                'profit_mean': float(v['profit_mean']),
                'profit_std': float(v['profit_std']),
            } for k, v in mc_p2.items()},
        }

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  全部求解完成! 耗时: {elapsed:.1f} 秒")
    print(f"  输出图表: output/figures/")
    print(f"{'=' * 60}")

    # Save results
    save_results(all_results, 'results.json')


if __name__ == '__main__':
    main()
