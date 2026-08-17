"""
多元线性回归完整实现
支持正规方程和梯度下降两种求解方法
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import warnings
import matplotlib.font_manager as fm

# ========== 中文字体设置 ==========
# Windows 系统常用路径，根据你的系统选择：
font_path = r'C:\Windows\Fonts\simhei.ttf'  # 黑体
# font_path = r'C:\Windows\Fonts\msyh.ttc'   # 微软雅黑（推荐）

try:
    chinese_font = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = chinese_font.get_name()
except:
    # 如果找不到文件，尝试用字体名称
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    
plt.rcParams['axes.unicode_minus'] = False

class MultipleLinearRegression:
    """
    多元线性回归实现（最小二乘法）
    支持正规方程求解和梯度下降两种训练方式
    梯度下降模式下自动进行特征标准化
    """

    def __init__(
        self,
        method: str = "normal",
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
        random_state: Optional[int] = None,
    ):
        """
        初始化模型

        Parameters:
        -----------
        method : str
            求解方法：'normal'（正规方程）或 'gradient'（梯度下降）
        learning_rate : float
            梯度下降的学习率
        max_iter : int
            梯度下降的最大迭代次数
        tol : float
            梯度下降的收敛阈值
        random_state : int or None
            随机种子（用于梯度下降初始化）
        """
        self.method = method
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.coef_ = None  # 回归系数（不含截距，基于原始特征尺度）
        self.intercept_ = None  # 截距（基于原始特征尺度）
        self.beta_ = None  # 完整参数向量 [intercept, coef1, coef2, ...]
        self.loss_history_ = []  # 梯度下降的损失历史
        self.n_features_ = None
        self.is_fitted = False

        # 标准化相关参数（仅梯度下降使用）
        self._X_mean = None
        self._X_std = None
        self._y_mean = None
        self._use_standardize = False

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """为设计矩阵添加截距列（全1列）"""
        return np.column_stack([np.ones(X.shape[0]), X])

    def _standardize(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        fit: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """标准化特征和目标值"""
        if fit:
            self._X_mean = np.mean(X, axis=0)
            self._X_std = np.std(X, axis=0)
            self._X_std[self._X_std == 0] = 1  # 避免除零
            if y is not None:
                self._y_mean = np.mean(y)

        X_scaled = (X - self._X_mean) / self._X_std
        if y is not None:
            y_scaled = y - self._y_mean if self._y_mean is not None else y
            return X_scaled, y_scaled
        return X_scaled, None

    def _normal_equation(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """正规方程求解：beta = (X^T X)^(-1) X^T y"""
        X_design = self._add_intercept(X)
        XtX = X_design.T @ X_design

        try:
            beta = np.linalg.pinv(XtX) @ X_design.T @ y
        except np.linalg.LinAlgError:
            warnings.warn("X^T X 是奇异矩阵，使用伪逆求解")
            beta = np.linalg.pinv(X_design) @ y

        return beta

    def _gradient_descent(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """批量梯度下降求解（使用标准化数据）"""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # 标准化数据以加速收敛
        X_scaled, y_scaled = self._standardize(X, y, fit=True)
        self._use_standardize = True

        X_design = self._add_intercept(X_scaled)
        m, n = X_design.shape

        # 初始化参数
        beta = np.random.randn(n) * 0.01

        self.loss_history_ = []

        for i in range(self.max_iter):
            y_pred = X_design @ beta

            # 计算损失（MSE）
            loss = np.mean((y_pred - y_scaled) ** 2)
            self.loss_history_.append(loss)

            # 检查发散
            if np.isnan(loss) or np.isinf(loss):
                raise RuntimeError(
                    f"梯度下降在第 {i + 1} 次迭代时发散，请尝试降低学习率"
                )

            # 计算梯度
            gradient = (2 / m) * X_design.T @ (y_pred - y_scaled)

            # 更新参数
            beta_new = beta - self.learning_rate * gradient

            # 检查收敛
            if np.linalg.norm(beta_new - beta) < self.tol:
                print(f"梯度下降在第 {i + 1} 次迭代后收敛")
                break

            beta = beta_new
        else:
            print(f"梯度下降达到最大迭代次数 {self.max_iter}")

        # 将标准化后的参数转换回原始尺度
        # y = y_mean + X_scaled @ beta[1:] + beta[0]
        # y = y_mean + (X - X_mean)/X_std @ beta[1:] + beta[0]
        # y = (y_mean + beta[0] - sum(beta[i]*X_mean[i]/X_std[i])) + sum(beta[i]/X_std[i] * X[i])
        intercept = (
            self._y_mean
            + beta[0]
            - np.sum(beta[1:] * self._X_mean / self._X_std)
        )
        coefs = beta[1:] / self._X_std

        return np.concatenate([[intercept], coefs])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultipleLinearRegression":
        """
        训练模型

        Parameters:
        -----------
        X : np.ndarray, shape (m, n)
            训练数据，m个样本，n个特征
        y : np.ndarray, shape (m,)
            目标值

        Returns:
        --------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_ = X.shape[1]

        if self.method == "normal":
            self.beta_ = self._normal_equation(X, y)
        elif self.method == "gradient":
            self.beta_ = self._gradient_descent(X, y)
        else:
            raise ValueError("method 必须是 'normal' 或 'gradient'")

        self.intercept_ = self.beta_[0]
        self.coef_ = self.beta_[1:]
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit()")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1 and self.n_features_ == 1:
            X = X.reshape(-1, 1)
        elif X.ndim == 1:
            X = X.reshape(1, -1)

        X_design = self._add_intercept(X)
        return X_design @ self.beta_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算 R^2 分数"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    def rmse(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算均方根误差"""
        y_pred = self.predict(X)
        return np.sqrt(np.mean((y - y_pred) ** 2))

    def mae(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算平均绝对误差"""
        y_pred = self.predict(X)
        return np.mean(np.abs(y - y_pred))

    def adjusted_r2(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算调整 R^2"""
        m = X.shape[0]
        n = self.n_features_
        r2 = self.score(X, y)
        return 1 - (1 - r2) * (m - 1) / (m - n - 1) if m > n + 1 else r2

    def summary(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> str:
        """生成模型摘要报告"""
        if feature_names is None:
            feature_names = [f"X{i + 1}" for i in range(self.n_features_)]

        r2 = self.score(X, y)
        adj_r2 = self.adjusted_r2(X, y)
        rmse_val = self.rmse(X, y)
        mae_val = self.mae(X, y)

        report = "=" * 60 + "\n"
        report += "           多元线性回归模型摘要\n"
        report += "=" * 60 + "\n\n"

        report += f"【模型信息】\n"
        report += f"  求解方法: {'正规方程' if self.method == 'normal' else '梯度下降'}\n"
        report += f"  样本数量: {X.shape[0]}\n"
        report += f"  特征数量: {self.n_features_}\n\n"

        report += "【回归系数】\n"
        report += f"  截距 (β₀): {self.intercept_: .6f}\n"
        for name, coef in zip(feature_names, self.coef_):
            report += f"  {name:8s} (β): {coef: .6f}\n"
        report += "\n"

        report += "【评估指标】\n"
        report += f"  R²        : {r2:.6f}\n"
        report += f"  调整 R²   : {adj_r2:.6f}\n"
        report += f"  RMSE      : {rmse_val:.6f}\n"
        report += f"  MAE       : {mae_val:.6f}\n"
        report += "=" * 60

        return report


# ==================== 可视化函数 ====================


def visualize_regression(
    model: MultipleLinearRegression,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 12),
):
    """
    可视化多元线性回归结果

    包含：
    1. 实际值 vs 预测值散点图
    2. 残差分布图
    3. 残差 vs 预测值图（检验同方差性）
    4. 各特征的回归系数条形图
    5. 梯度下降损失曲线（如适用）
    6. 各特征与目标值的关系（带回归线）
    """
    if not model.is_fitted:
        raise RuntimeError("模型尚未训练")

    y_pred = model.predict(X)
    residuals = y - y_pred

    if feature_names is None:
        feature_names = [f"X{i + 1}" for i in range(model.n_features_)]

    n_features = model.n_features_
    n_rows = 3
    n_cols = max(2, n_features)

    fig = plt.figure(figsize=figsize)

    # 1. 实际值 vs 预测值
    ax1 = plt.subplot(n_rows, n_cols, 1)
    ax1.scatter(
        y,
        y_pred,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
        c="steelblue",
        s=50,
    )
    min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="理想拟合线")
    ax1.set_xlabel("实际值", fontsize=11)
    ax1.set_ylabel("预测值", fontsize=11)
    ax1.set_title("实际值 vs 预测值", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 残差分布
    ax2 = plt.subplot(n_rows, n_cols, 2)
    ax2.hist(
        residuals,
        bins=30,
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
        density=True,
    )
    mu, sigma = np.mean(residuals), np.std(residuals)
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    y_norm = (
        1 / (sigma * np.sqrt(2 * np.pi))
    ) * np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2)
    ax2.plot(x_norm, y_norm, "r-", lw=2, label=f"N({mu:.3f}, {sigma**2:.3f})")
    ax2.set_xlabel("残差", fontsize=11)
    ax2.set_ylabel("密度", fontsize=11)
    ax2.set_title("残差分布直方图", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 残差 vs 预测值
    ax3 = plt.subplot(n_rows, n_cols, n_cols + 1)
    ax3.scatter(
        y_pred,
        residuals,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
        c="steelblue",
        s=50,
    )
    ax3.axhline(y=0, color="r", linestyle="--", lw=2)
    ax3.set_xlabel("预测值", fontsize=11)
    ax3.set_ylabel("残差", fontsize=11)
    ax3.set_title("残差 vs 预测值（同方差性检验）", fontsize=13, fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # 4. 回归系数条形图
    ax4 = plt.subplot(n_rows, n_cols, n_cols + 2)
    all_names = ["截距"] + feature_names
    all_coefs = [model.intercept_] + list(model.coef_)
    colors = ["coral"] + ["steelblue"] * len(feature_names)
    bars = ax4.barh(all_names, all_coefs, color=colors, edgecolor="black", alpha=0.8)
    ax4.set_xlabel("系数值", fontsize=11)
    ax4.set_title("回归系数", fontsize=13, fontweight="bold")
    ax4.axvline(x=0, color="black", linestyle="-", lw=0.5)
    ax4.grid(True, alpha=0.3, axis="x")
    for bar, coef in zip(bars, all_coefs):
        ax4.text(
            coef,
            bar.get_y() + bar.get_height() / 2,
            f"{coef:.3f}",
            va="center",
            ha="left" if coef >= 0 else "right",
            fontsize=9,
        )

    # 5. 梯度下降损失曲线
    ax5 = plt.subplot(n_rows, n_cols, 2 * n_cols + 1)
    if model.method == "gradient" and len(model.loss_history_) > 0:
        ax5.plot(model.loss_history_, color="steelblue", lw=2)
        ax5.set_xlabel("迭代次数", fontsize=11)
        ax5.set_ylabel("MSE 损失（标准化后）", fontsize=11)
        ax5.set_title("梯度下降损失曲线", fontsize=13, fontweight="bold")
        ax5.grid(True, alpha=0.3)
        ax5.set_yscale("log")
    else:
        ax5.text(
            0.5,
            0.5,
            "使用正规方程求解\n无损失曲线",
            ha="center",
            va="center",
            transform=ax5.transAxes,
            fontsize=12,
        )
        ax5.set_title("训练过程", fontsize=13, fontweight="bold")
        ax5.axis("off")

    # 6. 各特征与目标值的关系
    for i in range(min(n_features, n_cols - 1)):
        ax = plt.subplot(n_rows, n_cols, 2 * n_cols + 2 + i)
        ax.scatter(
            X[:, i],
            y,
            alpha=0.5,
            edgecolors="black",
            linewidth=0.5,
            c="steelblue",
            s=40,
            label="数据点",
        )
        x_range = np.linspace(X[:, i].min(), X[:, i].max(), 100)
        X_mean = np.mean(X, axis=0)
        X_line = np.tile(X_mean, (100, 1))
        X_line[:, i] = x_range
        y_line = model.predict(X_line)
        ax.plot(x_range, y_line, "r-", lw=2, label="回归线")
        ax.set_xlabel(feature_names[i], fontsize=11)
        ax.set_ylabel("y", fontsize=11)
        ax.set_title(f"{feature_names[i]} vs y", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "多元线性回归分析可视化", fontsize=16, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.show()


# ==================== 使用示例 ====================

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    # 生成模拟数据
    np.random.seed(42)
    m = 200
    n = 3
    true_beta = np.array([5.0, 2.5, -1.8, 3.0])

    X = np.random.randn(m, n)
    X[:, 0] = X[:, 0] * 2 + 10
    X[:, 1] = np.random.uniform(0, 5, m)
    X[:, 2] = np.random.poisson(3, m)

    y = true_beta[0] + X @ true_beta[1:] + np.random.normal(0, 2, m)

    feature_names = ["温度", "湿度", "风速"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 正规方程
    print("=" * 60)
    print("【正规方程求解】")
    print("=" * 60)
    model_normal = MultipleLinearRegression(method="normal")
    model_normal.fit(X_train, y_train)
    print(model_normal.summary(X_test, y_test, feature_names))

    # 梯度下降
    print("\n" + "=" * 60)
    print("【梯度下降求解】")
    print("=" * 60)
    model_gd = MultipleLinearRegression(
        method="gradient", learning_rate=0.05, max_iter=5000, random_state=42
    )
    model_gd.fit(X_train, y_train)
    print(model_gd.summary(X_test, y_test, feature_names))

    # 可视化
    visualize_regression(model_normal, X_test, y_test, feature_names)