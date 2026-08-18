"""
Problem 1: Sampling Inspection Scheme
======================================
Wald's Sequential Probability Ratio Test (SPRT) + Bayesian Optimal Stopping

数学模型:
  H₀: p ≤ p₀ (次品率不超过标称值)
  H₁: p ≥ p₁ (次品率超过标称值)

  Wald SPRT 边界:
    A = β/(1-α)  (下界, 接收H₀)
    B = (1-β)/α  (上界, 拒绝H₀)

  对数似然比:
    λ_n = Σ_{i=1}^n [X_i log(p₁/p₀) + (1-X_i) log((1-p₁)/(1-p₀))]
"""

import numpy as np
from scipy import stats, optimize
from scipy.special import betaln, digamma
import matplotlib.pyplot as plt
from src.plot_utils import savefig, new_figure, COLORS, OUTPUT_DIR


# ============================================================
# Part A: Wald SPRT
# ============================================================

class WaldSPRT:
    """Wald's Sequential Probability Ratio Test."""

    def __init__(self, p0, p1, alpha, beta):
        """
        Parameters
        ----------
        p0 : float — H₀ 标称次品率
        p1 : float — H₁ 备择次品率 (p1 > p0)
        alpha : float — Type I error (reject H₀ when true)
        beta : float — Type II error (accept H₀ when false)
        """
        self.p0 = p0
        self.p1 = p1
        self.alpha = alpha
        self.beta = beta

        # Decision boundaries (log scale)
        self.log_A = np.log(beta / (1 - alpha))       # lower: accept H₀
        self.log_B = np.log((1 - beta) / alpha)       # upper: reject H₀

        # Per-observation log-likelihood ratio coefficients
        self.llr_good = np.log((1 - p1) / (1 - p0))   # X_i = 0 (good)
        self.llr_bad = np.log(p1 / p0)                 # X_i = 1 (bad)

    def run_single(self, true_p, max_n=10000, seed=None):
        """Run one SPRT trial. Returns (decision, n_samples, history)."""
        rng = np.random.RandomState(seed)
        llr = 0.0
        history = [llr]

        for n in range(1, max_n + 1):
            x = 1 if rng.random() < true_p else 0
            llr += self.llr_bad if x == 1 else self.llr_good
            history.append(llr)

            if llr <= self.log_A:
                return 0, n, history  # Accept H₀
            if llr >= self.log_B:
                return 1, n, history  # Reject H₀

        # Truncation: decide based on final llr
        return 2, max_n, history  # Inconclusive (truncated)

    def operating_characteristic(self, p_grid=None):
        """Compute OC(p) = P(accept H₀ | true defect rate = p)."""
        if p_grid is None:
            p_grid = np.linspace(0.01, 0.30, 100)

        # Wald's approximation: OC(p) ≈ (B^h - 1) / (B^h - A^h)
        # where h solves: p*(p1/p0)^h + (1-p)*((1-p1)/(1-p0))^h = 1
        oc = np.zeros_like(p_grid)

        for i, p in enumerate(p_grid):
            if p == 0:
                oc[i] = 1.0
                continue
            if p == 1:
                oc[i] = 0.0
                continue

            # Solve for h: p*(p1/p0)^h + (1-p)*((1-p1)/(1-p0))^h = 1
            def f(h):
                if h == 0:
                    return 1.0 - 1.0
                return p * (self.p1 / self.p0) ** h + (1 - p) * ((1 - self.p1) / (1 - self.p0)) ** h - 1

            # h=0 always a root; find the non-zero root
            try:
                if p < self.p0:
                    # h > 0
                    bracket = [1e-6, 100]
                else:
                    bracket = [-100, -1e-6]

                h_sol = optimize.brentq(f, bracket[0], bracket[1], maxiter=200)
                A = np.exp(self.log_A)
                B = np.exp(self.log_B)
                oc[i] = (B ** h_sol - 1) / (B ** h_sol - A ** h_sol)
            except (ValueError, OverflowError, ZeroDivisionError):
                # Fallback
                if p <= self.p0:
                    oc[i] = 1.0 - self.alpha
                else:
                    oc[i] = self.beta

        return p_grid, np.clip(oc, 0, 1)

    def avg_sample_number(self, p_grid=None):
        """Compute ASN(p) = E[n | true defect rate = p]."""
        if p_grid is None:
            p_grid = np.linspace(0.01, 0.30, 100)

        asn = np.zeros_like(p_grid)
        for i, p in enumerate(p_grid):
            # E[Z] = expected per-observation LLR
            if p == 0:
                ez = self.llr_good
            elif p == 1:
                ez = self.llr_bad
            else:
                ez = p * self.llr_bad + (1 - p) * self.llr_good

            if abs(ez) < 1e-15:
                asn[i] = -self.log_A * self.log_B / (
                    p * self.llr_bad**2 + (1-p) * self.llr_good**2)
            else:
                oc_p = self.operating_characteristic(np.array([p]))[1][0]
                asn[i] = (oc_p * self.log_A + (1 - oc_p) * self.log_B) / ez

        return p_grid, np.maximum(asn, 1)

    def monte_carlo_oc_asn(self, p_list, n_trials=5000, seed=42):
        """Monte Carlo estimation of OC and ASN."""
        rng = np.random.RandomState(seed)
        oc_mc = np.zeros(len(p_list))
        asn_mc = np.zeros(len(p_list))

        for i, p in enumerate(p_list):
            accepts = 0
            total_n = 0.0
            for _ in range(n_trials):
                decision, n, _ = self.run_single(p, seed=rng.randint(0, 2**31))
                total_n += n
                if decision == 0:
                    accepts += 1
                elif decision == 2:  # inconclusive, count fractionally
                    accepts += 0.5
            oc_mc[i] = accepts / n_trials
            asn_mc[i] = total_n / n_trials

        return oc_mc, asn_mc


# ============================================================
# Part B: Bayesian Optimal Stopping
# ============================================================

class BayesianStopping:
    """
    Bayesian optimal stopping for acceptance sampling.

    State: Beta(α, β) posterior of defect rate p.
    At each step, choose: accept, reject, or sample one more.
    """

    def __init__(self, alpha_prior, beta_prior, cost_test,
                 loss_accept_bad, loss_reject_good, p_threshold,
                 max_samples=200):
        self.alpha0 = alpha_prior
        self.beta0 = beta_prior
        self.c = cost_test
        self.L_accept = loss_accept_bad   # Loss if we accept a batch with p > threshold
        self.L_reject = loss_reject_good  # Loss if we reject a batch with p ≤ threshold
        self.p0 = p_threshold
        self.N_max = max_samples

    def expected_loss_accept(self, alpha, beta):
        """Expected loss of accepting batch given Beta(α, β) posterior."""
        # P(p > p0 | α, β) = 1 - Beta_CDF(p0; α, β)
        prob_bad = 1.0 - stats.beta.cdf(self.p0, alpha, beta)
        return self.L_accept * prob_bad

    def expected_loss_reject(self, alpha, beta):
        """Expected loss of rejecting batch given Beta(α, β) posterior."""
        prob_good = stats.beta.cdf(self.p0, alpha, beta)
        return self.L_reject * prob_good

    def solve_dp(self, grid_size=100):
        """
        Solve the Bayesian optimal stopping DP via backward induction
        on a discretized (α, β) grid.

        Returns the value function V(α,β) and the optimal policy.
        """
        # We discretize the posterior mean and precision
        # State = (k, n) where k = #defects observed, n = total samples
        # α = α0 + k, β = β0 + (n-k)

        # Value function: V[k, n] for k ∈ [0, n], n ∈ [0, N_max]
        V = np.zeros((self.N_max + 1, self.N_max + 1))
        policy = np.zeros((self.N_max + 1, self.N_max + 1), dtype=int)
        # policy: 0=accept, 1=reject, 2=continue

        # Terminal: at n = N_max, must decide
        for n in range(self.N_max + 1):
            for k in range(n + 1):
                alpha = self.alpha0 + k
                beta = self.beta0 + (n - k)
                loss_acc = self.expected_loss_accept(alpha, beta)
                loss_rej = self.expected_loss_reject(alpha, beta)
                if loss_acc < loss_rej:
                    V[k, n] = loss_acc
                    policy[k, n] = 0
                else:
                    V[k, n] = loss_rej
                    policy[k, n] = 1

        # Backward induction
        for n in range(self.N_max - 1, -1, -1):
            for k in range(n + 1):
                alpha = self.alpha0 + k
                beta = self.beta0 + (n - k)

                # Current posterior mean
                p_mean = alpha / (alpha + beta)

                # Expected future value if we sample
                # Next: (k+1, n+1) with prob p_mean, (k, n+1) with prob 1-p_mean
                v_sample = (self.c +
                            p_mean * V[k + 1, n + 1] +
                            (1 - p_mean) * V[k, n + 1])

                loss_acc = self.expected_loss_accept(alpha, beta)
                loss_rej = self.expected_loss_reject(alpha, beta)

                best = min(loss_acc, loss_rej, v_sample)
                V[k, n] = best

                if best == loss_acc:
                    policy[k, n] = 0
                elif best == loss_rej:
                    policy[k, n] = 1
                else:
                    policy[k, n] = 2

        return V, policy

    def get_stopping_boundary(self, policy):
        """Extract the stopping boundary from the policy grid."""
        # For each n, find the range of k where policy says "continue"
        boundaries_accept = []  # max k to accept
        boundaries_reject = []  # min k to reject

        for n in range(self.N_max + 1):
            k_accept_max = -1
            k_reject_min = n + 1
            for k in range(n + 1):
                if policy[k, n] == 0:  # accept
                    k_accept_max = max(k_accept_max, k)
                if policy[k, n] == 1:  # reject
                    k_reject_min = min(k_reject_min, k)
            boundaries_accept.append(k_accept_max)
            boundaries_reject.append(k_reject_min)

        return boundaries_accept, boundaries_reject


# ============================================================
# Part C: Solve Problem 1
# ============================================================

def solve_problem1():
    """Solve Problem 1: Design sampling inspection scheme."""
    print("=" * 60)
    print("问题1: 抽样检测方案设计")
    print("=" * 60)

    p0 = 0.10  # 标称值
    alpha = 0.05  # 95%信度
    beta = 0.10   # 90%信度
    p1 = 0.15     # 备择值

    sprt = WaldSPRT(p0, p1, alpha, beta)

    print(f"\nSPRT 参数:")
    print(f"  H₀: p ≤ {p0}, H₁: p ≥ {p1}")
    print(f"  α = {alpha}, β = {beta}")
    print(f"  log(A) = {sprt.log_A:.4f}, log(B) = {sprt.log_B:.4f}")

    # --- Case (1): 95%信度拒绝 ---
    print(f"\n--- 情形(1): {1-alpha:.0%}信度下认定次品率超过{p0:.0%}, 则拒收 ---")
    # Simulate at p = p0 (should mostly accept)
    decisions_at_p0 = []
    ns_at_p0 = []
    for i in range(1000):
        d, n, _ = sprt.run_single(p0, seed=i)
        decisions_at_p0.append(d)
        ns_at_p0.append(n)
    reject_rate_at_p0 = sum(1 for d in decisions_at_p0 if d == 1) / len(decisions_at_p0)
    avg_n_p0 = np.mean(ns_at_p0)
    print(f"  在 p={p0} 处: 误拒率 = {reject_rate_at_p0:.4f}, 平均样本量 = {avg_n_p0:.1f}")

    # Simulate at p = p1 (should mostly reject)
    decisions_at_p1 = []
    ns_at_p1 = []
    for i in range(1000):
        d, n, _ = sprt.run_single(p1, seed=i+10000)
        decisions_at_p1.append(d)
        ns_at_p1.append(n)
    reject_rate_at_p1 = sum(1 for d in decisions_at_p1 if d == 1) / len(decisions_at_p1)
    avg_n_p1 = np.mean(ns_at_p1)
    print(f"  在 p={p1} 处: 拒收率 = {reject_rate_at_p1:.4f}, 平均样本量 = {avg_n_p1:.1f}")

    # --- Case (2): 90%信度接收 ---
    print(f"\n--- 情形(2): {1-beta:.0%}信度下认定次品率不超过{p0:.0%}, 则接收 ---")
    # At p = p0, acceptance rate should be ≥ 90%
    accept_rate_p0 = 1 - reject_rate_at_p0
    print(f"  在 p={p0} 处: 接收率 = {accept_rate_p0:.4f} (要求 ≥ {1-beta:.0%})")

    # --- OC and ASN Curves ---
    print("\n--- OC曲线与ASN曲线 ---")
    p_grid = np.linspace(0.02, 0.28, 80)
    p_oc, oc = sprt.operating_characteristic(p_grid)
    p_asn, asn = sprt.avg_sample_number(p_grid)

    # Plot OC curve
    fig, ax = new_figure((10, 5), title="Wald SPRT 操作特性 (OC) 曲线")
    ax.plot(p_oc, oc, 'b-', linewidth=2, label=f'OC(p)')
    ax.axvline(x=p0, color='g', linestyle='--', alpha=0.7, label=f'标称值 p0={p0}')
    ax.axvline(x=p1, color='r', linestyle='--', alpha=0.7, label=f'备择值 p1={p1}')
    ax.axhline(y=1-alpha, color='g', linestyle=':', alpha=0.5)
    ax.axhline(y=beta, color='r', linestyle=':', alpha=0.5)
    ax.fill_between([p0, p1], 0, 1, alpha=0.1, color='gray', label='无差别区间')
    ax.set_xlabel('真实次品率 p')
    ax.set_ylabel('接收概率 OC(p)')
    ax.legend(loc='upper right')
    ax.set_xlim([p_grid[0], p_grid[-1]])
    ax.set_ylim([0, 1.05])
    savefig(fig, 'problem1_oc_curve.png')
    plt.close(fig)

    # Plot ASN curve
    fig, ax = new_figure((10, 5), title="Wald SPRT 平均样本量 (ASN) 曲线")
    ax.plot(p_asn, asn, 'b-', linewidth=2, label='ASN(p)')
    ax.axvline(x=p0, color='g', linestyle='--', alpha=0.7, label=f'p0={p0}')
    ax.axvline(x=p1, color='r', linestyle='--', alpha=0.7, label=f'p1={p1}')
    ax.set_xlabel('真实次品率 p')
    ax.set_ylabel('平均样本量 E[N]')
    ax.legend(loc='upper center')
    ax.set_xlim([p_grid[0], p_grid[-1]])
    savefig(fig, 'problem1_asn_curve.png')
    plt.close(fig)

    # --- One example SPRT trajectory ---
    fig, ax = new_figure((10, 4), title="SPRT 单次抽样轨迹示例")
    for seed, label, color in [(42, 'p=0.08 (<p0) -> Accept', COLORS[0]),
                                (43, 'p=0.10 (=p0) -> Accept', COLORS[1]),
                                (99, 'p=0.15 (=p1) -> Reject', COLORS[2]),
                                (100, 'p=0.20 (>p1) -> Reject', COLORS[3])]:
        p_true = 0.08 if '<p0' in label else (0.10 if '=p0' in label else
                                               (0.15 if '=p1' in label else 0.20))
        d, n, hist = sprt.run_single(p_true, seed=seed)
        ax.plot(range(len(hist)), hist, color=color, linewidth=1.5, alpha=0.8, label=label)

    ax.axhline(y=sprt.log_A, color='green', linestyle='-', linewidth=1.5, alpha=0.5, label=f'下界 ln(A)={sprt.log_A:.2f}')
    ax.axhline(y=sprt.log_B, color='red', linestyle='-', linewidth=1.5, alpha=0.5, label=f'上界 ln(B)={sprt.log_B:.2f}')
    ax.set_xlabel('抽样数 n')
    ax.set_ylabel('对数似然比 ln(lambda_n)')
    ax.legend(loc='best', fontsize=8)
    savefig(fig, 'problem1_sprt_trajectories.png')
    plt.close(fig)

    # --- Bayesian Optimal Stopping ---
    print("\n--- 贝叶斯最优停止 ---")
    bayes = BayesianStopping(
        alpha_prior=1.0, beta_prior=9.0,
        cost_test=0.01,   # relative to loss_accept_bad
        loss_accept_bad=1.0,
        loss_reject_good=0.5,
        p_threshold=p0,
        max_samples=100
    )
    V, policy = bayes.solve_dp()
    bound_acc, bound_rej = bayes.get_stopping_boundary(policy)

    # Plot Bayesian stopping boundaries
    fig, ax = new_figure((10, 6), title="贝叶斯最优停止边界")
    ns = np.arange(bayes.N_max + 1)
    ax.fill_between(ns, bound_acc, bound_rej, alpha=0.2, color='orange', label='继续抽样区域')
    ax.step(ns, bound_acc, 'g-', where='post', linewidth=1.5, label='接收边界 (max k)')
    ax.step(ns, bound_rej, 'r-', where='post', linewidth=1.5, label='拒收边界 (min k)')

    # Overlay: k/n = p0 line
    ax.plot(ns[1:], p0 * ns[1:], 'k--', alpha=0.4, label=f'k/n = {p0}')
    ax.set_xlabel('抽样总数 n')
    ax.set_ylabel('不合格品数 k')
    ax.legend(loc='upper left')
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 35])
    savefig(fig, 'problem1_bayesian_boundary.png')
    plt.close(fig)

    # Summary results
    results = {
        'sprt': sprt,
        'p0': p0, 'p1': p1, 'alpha': alpha, 'beta': beta,
        'log_A': sprt.log_A, 'log_B': sprt.log_B,
        'reject_rate_at_p0': reject_rate_at_p0,
        'reject_rate_at_p1': reject_rate_at_p1,
        'avg_n_p0': avg_n_p0, 'avg_n_p1': avg_n_p1,
        'p_grid': p_grid, 'oc': oc, 'asn': asn,
    }

    print(f"\n问题1 结论:")
    print(f"  情形(1): 在 p={p0} 处实际误拒率={reject_rate_at_p0:.4f} ≤ α={alpha}")
    print(f"  情形(2): 在 p={p0} 处实际接收率={1-reject_rate_at_p0:.4f} ≥ {1-beta}")
    print(f"  ASN(p₀) = {avg_n_p0:.1f}, ASN(p₁) = {avg_n_p1:.1f}")

    return results


if __name__ == '__main__':
    solve_problem1()
