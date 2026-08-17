"""
一元线性回归模型 - 完整封装类
================================
支持: 拟合、预测、评估、可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class LinearRegression:
    """
    一元线性回归模型 (最小二乘法)

    模型: y = beta0 + beta1 * x + epsilon

    Attributes:
        beta0 (float): 截距
        beta1 (float): 斜率
        r_squared (float): 决定系数 R²
        mse (float): 均方误差
        rmse (float): 均方根误差
        sse (float): 残差平方和
        sst (float): 总平方和
    """

    def __init__(self):
        self.beta0 = None
        self.beta1 = None
        self.r_squared = None
        self.mse = None
        self.rmse = None
        self.sse = None
        self.sst = None
        self.x_mean = None
        self.y_mean = None
        self.n = None
        self.x_train = None
        self.y_train = None
        self.y_pred_train = None
        self.residuals = None

    def fit(self, x, y):
        """
        使用最小二乘法拟合模型

        Parameters:
            x (array-like): 自变量
            y (array-like): 因变量

        Returns:
            self: 返回模型实例，支持链式调用
        """
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)

        self.n = len(x)
        self.x_mean = np.mean(x)
        self.y_mean = np.mean(y)

        # 最小二乘法计算参数
        numerator = np.sum((x - self.x_mean) * (y - self.y_mean))
        denominator = np.sum((x - self.x_mean) ** 2)

        if denominator == 0:
            raise ValueError("自变量 x 的方差为0，无法拟合")

        self.beta1 = numerator / denominator
        self.beta0 = self.y_mean - self.beta1 * self.x_mean

        # 计算预测值和残差
        y_pred = self.predict(x)
        residuals = y - y_pred

        # 评估指标
        self.sse = np.sum(residuals  **2)
        self.sst = np.sum((y - self.y_mean) ** 2)
        self.r_squared = 1 - self.sse / self.sst if self.sst != 0 else 1.0
        self.mse = self.sse / self.n
        self.rmse = np.sqrt(self.mse)

        # 保存训练数据
        self.x_train = x
        self.y_train = y
        self.y_pred_train = y_pred
        self.residuals = residuals

        return self

    def predict(self, x):
        """
        使用拟合的模型进行预测

        Parameters:
            x (array-like): 自变量

        Returns:
            numpy.ndarray: 预测值
        """
        if self.beta0 is None or self.beta1 is None:
            raise ValueError("模型尚未拟合，请先调用 fit()")

        x = np.array(x, dtype=float)
        return self.beta0 + self.beta1 * x

    def score(self, x, y):
        """
        计算给定数据的 R² 分数

        Parameters:
            x (array-like): 自变量
            y (array-like): 因变量

        Returns:
            float: R² 值
        """
        y = np.array(y, dtype=float)
        y_pred = self.predict(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else 1.0

    def summary(self):
        """打印模型摘要信息"""
        if self.beta0 is None:
            print("模型尚未拟合")
            return

        print("=" * 50)
        print("         一元线性回归模型摘要")
        print("=" * 50)
        print(f"  回归方程: y = {self.beta0:.4f} + {self.beta1:.4f} * x")
        print(f"  样本量 n: {self.n}")
        print("-" * 50)
        print("  参数估计:")
        print(f"    截距 β₀ = {self.beta0:.6f}")
        print(f"    斜率 β₁ = {self.beta1:.6f}")
        print("-" * 50)
        print("  模型评估:")
        print(f"    R² (决定系数) = {self.r_squared:.6f}")
        print(f"    MSE (均方误差) = {self.mse:.6f}")
        print(f"    RMSE (均方根误差) = {self.rmse:.6f}")
        print(f"    SSE (残差平方和) = {self.sse:.6f}")
        print(f"    SST (总平方和) = {self.sst:.6f}")
        print("=" * 50)

    def plot(self, figsize=(14, 12), save_path=None):
        """
        绘制完整的回归分析可视化

        Parameters:
            figsize (tuple): 图像尺寸
            save_path (str): 保存路径，None 则不保存
        """
        if self.beta0 is None:
            raise ValueError("模型尚未拟合，请先调用 fit()")

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('一元线性回归分析可视化', fontsize=16, fontweight='bold', y=1.02)

        color_data = '#2E86AB'
        color_line = '#E94F37'
        color_residual = '#F18F01'
        color_bg = '#F7F7F7'

        # ① 散点图与回归线
        ax1 = axes[0, 0]
        ax1.set_facecolor(color_bg)
        ax1.scatter(self.x_train, self.y_train, c=color_data, s=60, alpha=0.7,
                    edgecolors='white', linewidth=0.5, zorder=3, label='观测数据')
        x_line = np.linspace(self.x_train.min() - 1, self.x_train.max() + 1, 200)
        y_line = self.predict(x_line)
        ax1.plot(x_line, y_line, color=color_line, linewidth=2.5, zorder=2, label='回归线')
        ax1.axhline(y=self.y_mean, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax1.axvline(x=self.x_mean, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax1.text(0.05, 0.95, f'y = {self.beta0:.3f} + {self.beta1:.3f}x\nR² = {self.r_squared:.4f}',
                 transform=ax1.transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', edgecolor=color_line, alpha=0.9))
        ax1.set_xlabel('x (自变量)')
        ax1.set_ylabel('y (因变量)')
        ax1.set_title('① 散点图与回归线', fontweight='bold')
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # ② 残差图
        ax2 = axes[0, 1]
        ax2.set_facecolor(color_bg)
        ax2.scatter(self.y_pred_train, self.residuals, c=color_residual, s=50, alpha=0.7,
                    edgecolors='white', linewidth=0.5, zorder=3)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5, zorder=2)
        res_std = np.std(self.residuals)
        ax2.axhline(y=res_std, color='green', linestyle=':', linewidth=1, alpha=0.5)
        ax2.axhline(y=-res_std, color='green', linestyle=':', linewidth=1, alpha=0.5)
        ax2.set_xlabel('预测值 ŷ')
        ax2.set_ylabel('残差 (y - ŷ)')
        ax2.set_title('② 残差图 (检验同方差性)', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # ③ 实际值 vs 预测值
        ax3 = axes[1, 0]
        ax3.set_facecolor(color_bg)
        ax3.scatter(self.y_pred_train, self.y_train, c='#6A994E', s=50, alpha=0.7,
                    edgecolors='white', linewidth=0.5, zorder=3, label='数据点')
        min_val = min(self.y_train.min(), self.y_pred_train.min())
        max_val = max(self.y_train.max(), self.y_pred_train.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测线')
        ax3.set_xlabel('预测值 ŷ')
        ax3.set_ylabel('实际值 y')
        ax3.set_title('③ 实际值 vs 预测值', fontweight='bold')
        ax3.legend(loc='upper left', fontsize=9)
        ax3.grid(True, alpha=0.3)

        # ④ 残差分布
        ax4 = axes[1, 1]
        ax4.set_facecolor(color_bg)
        ax4.hist(self.residuals, bins=15, color=color_residual, edgecolor='white',
                 alpha=0.7, density=True, label='残差分布')
        mu, sigma = stats.norm.fit(self.residuals)
        x_norm = np.linspace(self.residuals.min() - 1, self.residuals.max() + 1, 200)
        ax4.plot(x_norm, stats.norm.pdf(x_norm, mu, sigma), 'r-', linewidth=2,
                 label=f'正态拟合 (μ={mu:.2f}, σ={sigma:.2f})')
        ax4.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax4.set_xlabel('残差')
        ax4.set_ylabel('概率密度')
        ax4.set_title('④ 残差分布 (检验正态性)', fontweight='bold')
        ax4.legend(loc='upper right', fontsize=9)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"图表已保存至: {save_path}")

        plt.show()

    def __repr__(self):
        if self.beta0 is not None:
            return f"LinearRegression(beta0={self.beta0:.4f}, beta1={self.beta1:.4f}, R²={self.r_squared:.4f})"
        return "LinearRegression(未拟合)"


# ============================================
# 使用示例
# ============================================
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n = 50
    x = np.linspace(0, 20, n)
    y = 10 + 2.5 * x + np.random.normal(0, 5, n)

    # 创建模型并拟合
    model = LinearRegression()
    model.fit(x, y)

    # 查看摘要
    model.summary()

    # 预测
    x_new = np.array([5, 10, 15])
    y_pred = model.predict(x_new)
    print(f"\n预测结果: {y_pred}")

    # 可视化
    model.plot(save_path="linear_regression.png")