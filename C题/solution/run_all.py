"""
Main entry point for 2024 CUMCM Problem C solution.
Runs all three problems and generates all visualizations.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import preprocess_all
from visualization import run_all_visualizations
from config import OUTPUT_DIR


def main():
    print("=" * 70)
    print("2024 CUMCM Problem C: Crop Planting Strategy Optimization")
    print("=" * 70)

    # ─── Step 1: Data Preprocessing ───
    print("\n" + "=" * 60)
    print("STEP 1: Data Preprocessing")
    print("=" * 60)
    data = preprocess_all()

    # ─── Step 2: Problem 1 - Deterministic Optimization ───
    print("\n" + "=" * 60)
    print("STEP 2: Problem 1 - Deterministic Optimization")
    print("=" * 60)
    try:
        from problem1 import run_problem1
        results_p1 = run_problem1(data)
        print("\nProblem 1 complete. Results exported to result1_1.xlsx and result1_2.xlsx")
    except Exception as e:
        print(f"Problem 1 skipped (error: {e})")
        results_p1 = None

    # ─── Step 3: Problem 2 - Robust Optimization ───
    print("\n" + "=" * 60)
    print("STEP 3: Problem 2 - Robust Optimization")
    print("=" * 60)
    try:
        from problem2 import run_problem2
        results_p2 = run_problem2(data)
        print("\nProblem 2 complete.")
    except Exception as e:
        print(f"Problem 2 skipped (error: {e})")
        results_p2 = None

    # ─── Step 4: Problem 3 - Correlation & Substitution ───
    print("\n" + "=" * 60)
    print("STEP 4: Problem 3 - Correlation & Substitution")
    print("=" * 60)
    try:
        from problem3 import run_problem3, compare_with_problem2
        results_p3 = run_problem3(data)
        if results_p2:
            comparison = compare_with_problem2(results_p3, results_p2)
        print("\nProblem 3 complete.")
    except Exception as e:
        print(f"Problem 3 skipped (error: {e})")
        results_p3 = None

    # ─── Step 5: Generate All Visualizations ───
    print("\n" + "=" * 60)
    print("STEP 5: Generate Visualizations")
    print("=" * 60)
    run_all_visualizations(data, results_p1, results_p2, results_p3)

    # ─── Summary ───
    print("\n" + "=" * 70)
    print("SOLUTION COMPLETE")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  Figures:     {OUTPUT_DIR}/")
    print(f"  Results P1:  result1_1.xlsx, result1_2.xlsx")
    print(f"  Results P2:  result2.xlsx")
    print(f"  Paper:       paper.tex")
    print("\nTo compile the paper:")
    print(f"  cd {os.path.dirname(os.path.abspath(__file__))}")
    print("  xelatex paper.tex")
    print("  xelatex paper.tex  # (run twice for cross-references)")

    return data, results_p1, results_p2, results_p3


if __name__ == '__main__':
    main()
