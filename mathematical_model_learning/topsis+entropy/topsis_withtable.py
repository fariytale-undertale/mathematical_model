"""
TOPSIS 多属性决策分析系统（完整版）
=====================================
支持：效益型、成本型、中间型、区间型

使用方法:
    from topsis_module import TOPSIS, entropy_weight, transform_to_benefit, topsis_complete

    # 创建模型
    model = TOPSIS()

    # 拟合数据
    model.fit(
        decision_matrix, 
        criteria_types,
        target_values=target_values,
        intervals=intervals,
        criteria_names=criteria_names,
        alternative_names=alternative_names
    )

    # 查看结果
    model.summary()

    # 获取最优方案
    best_name, best_score = model.get_best()

模块导出:
    - TOPSIS: 主分析类
    - entropy_weight: 熵权法计算函数
    - transform_to_benefit: 指标转换函数
    - topsis_complete: 完整 TOPSIS 计算函数
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import FancyBboxPatch

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


__all__ = [
    'entropy_weight',
    'transform_to_benefit', 
    'topsis_complete',
    'TOPSIS',
]


# ==================== 核心算法 ====================

def entropy_weight(X):
    """熵权法计算客观权重"""
    X_norm = np.zeros_like(X, dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        X_norm[:, j] = (col - col.min()) / (col.max() - col.min() + 1e-10)

    p = X_norm / (X_norm.sum(axis=0) + 1e-10)
    p = np.clip(p, 1e-10, 1)

    k = 1 / np.log(X.shape[0])
    entropy = -k * np.sum(p * np.log(p), axis=0)

    d = 1 - entropy
    weights = d / d.sum()

    return weights


def transform_to_benefit(X, criteria_types, target_values=None, intervals=None):
    """
    将所有指标统一转换为效益型（越大越好）

    按照标准公式处理：
    - 效益型: x̂ = (x-x_min)/(x_max-x_min)
    - 成本型: x̂ = (x_max-x)/(x_max-x_min)
    - 中间型: M = max{|xi - x_best|}, x̂ = 1 - |xi - x_best| / M
    - 区间型: M = max{a - min{xi}, max{xi} - b}, 分段线性转换

    参数:
        X: 原始决策矩阵
        criteria_types: 指标类型列表
            1 = 效益型（越大越好）
            -1 = 成本型（越小越好）
            0 = 中间型（越接近target越好）
            2 = 区间型（越落在interval内越好）
        target_values: 中间型指标的目标值列表（仅中间型需要）
        intervals: 区间型指标的区间列表，如 [(a1,b1), (a2,b2), ...]

    返回:
        X_transformed: 转换后的效益型矩阵
    """
    X = np.array(X, dtype=float)
    m, n = X.shape
    X_transformed = X.copy()

    for j in range(n):
        col = X[:, j]

        if criteria_types[j] == 1:  # 效益型：保持不变
            col_max = col.max()
            col_min = col.min()
            X_transformed[:, j] = (col - col_min)/(col_max - col_min)

        elif criteria_types[j] == -1:  # 成本型：x̂ = max - x
            col_max = col.max()
            col_min = col.min()
            X_transformed[:, j] = (col_max - col)/(col_max - col_min)

        elif criteria_types[j] == 0:  # 中间型
            if target_values is None or target_values[j] is None:
                raise ValueError(f"中间型指标 {j} 需要提供 target_values")
            x_best = target_values[j]

            M = np.max(np.abs(col - x_best))
            if M < 1e-10:
                X_transformed[:, j] = 1.0  # 所有值都等于目标值，全部最优
            else:
                X_transformed[:, j] = 1 - np.abs(col - x_best) / M

        elif criteria_types[j] == 2:  # 区间型
            if intervals is None or intervals[j] is None:
                raise ValueError(f"区间型指标 {j} 需要提供 intervals")
            a, b = intervals[j]

            M = max(a - col.min(), col.max() - b)
            if M < 1e-10:
                M = 1e-10  # 避免除零

            for i in range(m):
                xi = col[i]
                if xi < a:
                    X_transformed[i, j] = 1 - (a - xi) / M
                elif a <= xi <= b:
                    X_transformed[i, j] = 1.0
                else:  # xi > b
                    X_transformed[i, j] = 1 - (xi - b) / M

    return X_transformed


def topsis_complete(decision_matrix, criteria_types, target_values=None, 
                    intervals=None, use_entropy_weight=True, weights=None):
    """
    完整版 TOPSIS（支持四种指标类型）

    参数:
        decision_matrix: 决策矩阵 (m×n)
        criteria_types: 指标类型列表
            1 = 效益型, -1 = 成本型, 0 = 中间型, 2 = 区间型
        target_values: 中间型指标目标值列表
        intervals: 区间型指标区间列表
        use_entropy_weight: 是否使用熵权法
        weights: 手动权重
    """
    X = np.array(decision_matrix, dtype=float)
    n = X.shape[1]

    # 统一转换为效益型
    X_transformed = transform_to_benefit(X, criteria_types, target_values, intervals)

    # 确定权重
    if use_entropy_weight:
        w = entropy_weight(X_transformed)
    else:
        w = np.array(weights)

    # 对矩阵进行向量归一化
    norm = np.sqrt(np.sum(X_transformed**2, axis=0))
    R = X_transformed / norm
    # 防止除零
    norm[norm == 0] = 1e-10
    R = X_transformed / norm

    # 4. 加权标准化 
    V = R * w  # 广播乘法，简洁且正确

    # 确定正理想解和负理想解
    V_plus = np.zeros(n)
    V_minus = np.zeros(n)

    for j in range(n):
        if criteria_types[j] == 1:  # 效益型
            V_plus[j] = np.max(V[:, j])
            V_minus[j] = np.min(V[:, j])
        elif criteria_types[j] == -1:  # 成本型
            V_plus[j] = np.min(V[:, j])
            V_minus[j] = np.max(V[:, j])
        elif criteria_types[j] == 0:  # 中间型
            # 转换后越大越好，所以正理想解是最大值
            V_plus[j] = np.max(V[:, j])
            V_minus[j] = np.min(V[:, j])
        elif criteria_types[j] == 2:  # 区间型
            # 转换后越大越好
            V_plus[j] = np.max(V[:, j])
            V_minus[j] = np.min(V[:, j])

    # 计算距离
    D_plus = np.sqrt(np.sum((V - V_plus)**2, axis=1))
    D_minus = np.sqrt(np.sum((V - V_minus)**2, axis=1))

    # 贴近度
    scores = D_minus / (D_plus + D_minus)
    rankings = np.argsort(scores)[::-1]

    return scores, rankings, w, V_plus, V_minus, R, V, D_plus, D_minus, X_transformed


# ==================== 封装类 ====================

class TOPSIS:
    """
    TOPSIS 多属性决策分析类（完整版）

    支持四种指标类型：
        1  = 效益型（越大越好）
        -1 = 成本型（越小越好）
        0  = 中间型（越接近目标值越好）
        2  = 区间型（越落在区间内越好）
    """

    def __init__(self):
        self.scores = None
        self.rankings = None
        self.weights = None
        self.V_plus = None
        self.V_minus = None
        self.R = None
        self.V = None
        self.D_plus = None
        self.D_minus = None
        self.X_transformed = None
        self.df_result = None
        self.criteria_names = None
        self.alternative_names = None
        self.decision_matrix = None
        self.criteria_types = None
        self.target_values = None
        self.intervals = None

    def fit(self, X, criteria_types, target_values=None, intervals=None,
            criteria_names=None, alternative_names=None, weights=None):
        """
        执行 TOPSIS 分析

        参数:
            X: 决策矩阵
            criteria_types: 指标类型 [1, -1, 0, 2, ...]
            target_values: 中间型目标值（类型为0时需要）
            intervals: 区间型区间（类型为2时需要），如 [(a,b), None, ...]
            criteria_names: 指标名称
            alternative_names: 方案名称
            weights: 手动权重
        """
        self.decision_matrix = np.array(X, dtype=float)
        self.criteria_types = criteria_types
        self.target_values = target_values
        self.intervals = intervals
        self.criteria_names = criteria_names or [f"指标{i+1}" for i in range(len(criteria_types))]
        self.alternative_names = alternative_names or [f"方案{i+1}" for i in range(len(X))]

        # 检查参数
        for j, t in enumerate(criteria_types):
            if t == 0 and (target_values is None or target_values[j] is None):
                raise ValueError(f"指标 {j} ({self.criteria_names[j]}) 为中间型，需要提供 target_values")
            if t == 2 and (intervals is None or intervals[j] is None):
                raise ValueError(f"指标 {j} ({self.criteria_names[j]}) 为区间型，需要提供 intervals")

        self.scores, self.rankings, self.weights, self.V_plus, self.V_minus, \
        self.R, self.V, self.D_plus, self.D_minus, self.X_transformed = topsis_complete(
            X, criteria_types, target_values, intervals,
            use_entropy_weight=(weights is None), weights=weights
        )

        # 构建结果 DataFrame
        self.df_result = pd.DataFrame({
            '方案': self.alternative_names,
            'D+ (到正理想解距离)': self.D_plus.round(4),
            'D- (到负理想解距离)': self.D_minus.round(4),
            '贴近度 C*': self.scores.round(4),
            '排名': [0] * len(self.scores)
        })

        for rank, idx in enumerate(self.rankings, 1):
            self.df_result.loc[idx, '排名'] = rank

        self.df_result = self.df_result.sort_values('排名').reset_index(drop=True)

        return self

    def get_best(self):
        """获取最优方案"""
        best_idx = self.rankings[0]
        return self.alternative_names[best_idx], self.scores[best_idx]

    def summary(self):
        """打印汇总结果"""
        print("=" * 70)
        print("TOPSIS 评价结果汇总")
        print("=" * 70)

        # 打印指标类型说明
        print("\n【指标类型说明】")
        for j, t in enumerate(self.criteria_types):
            name = self.criteria_names[j]
            if t == 1:
                type_str = "效益型（越大越好）"
            elif t == -1:
                type_str = "成本型（越小越好）"
            elif t == 0:
                type_str = f"中间型（目标值={self.target_values[j]}）"
            elif t == 2:
                type_str = f"区间型（区间={self.intervals[j]}）"
            print(f"  {name}: {type_str}")

        print(f"\n【权重】")
        for name, w in zip(self.criteria_names, self.weights):
            print(f"  {name}: {w:.4f} ({w*100:.2f}%)")

        # ========== 新增：打印各指标归一化后的数据 ==========
        print(f"\n【指标归一化数据（转换后，越大越好）】")
        df_transformed = pd.DataFrame(
            self.X_transformed,
            columns=self.criteria_names,
            index=self.alternative_names
        )
        print(df_transformed.round(4).to_string())

        print(f"\n【向量归一化矩阵 R】")
        df_R = pd.DataFrame(
            np.array(self.R),
            columns=self.criteria_names,
            index=self.alternative_names
        )
        print(df_R.round(4).to_string())

        print(f"\n【加权标准化矩阵 V】")
        df_V = pd.DataFrame(
            np.array(self.V),
            columns=self.criteria_names,
            index=self.alternative_names
        )
        print(df_V.round(4).to_string())
        # =====================================================

        # 打印归一化后的数据
        print(f"\n【归一化后的决策矩阵（转换后全部为效益型）】")
        df_transformed = pd.DataFrame(
            self.X_transformed,
            columns=self.criteria_names,
            index=self.alternative_names
        )
        print(df_transformed.round(4).to_string())

        print(f"\n【结果】")
        print(self.df_result.to_string(index=False))
        print("=" * 70)
        best_name, best_score = self.get_best()
        print(f"最优方案: {best_name} (贴近度: {best_score:.4f})")
        print("=" * 70)

    # ==================== 可视化方法 ====================

    def plot_bar(self, figsize=(10, 6), save_path=None):
        """绘制得分柱状图"""
        fig, ax = plt.subplots(figsize=figsize)

        df_sorted = self.df_result.sort_values('贴近度 C*', ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_sorted)))

        bars = ax.barh(df_sorted['方案'], df_sorted['贴近度 C*'], 
                       color=colors, edgecolor='black', linewidth=0.5)

        for bar, val in zip(bars, df_sorted['贴近度 C*']):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.4f}', va='center', fontsize=10)

        ax.set_xlabel('贴近度 C*', fontsize=12)
        ax.set_title('TOPSIS 各方案贴近度得分', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.1)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return fig

    def plot_radar(self, figsize=(8, 8), save_path=None):
        """绘制雷达图"""
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

        angles = np.linspace(0, 2 * np.pi, len(self.criteria_names), endpoint=False).tolist()
        angles += angles[:1]

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.alternative_names)))

        for i, name in enumerate(self.alternative_names):
            values = self.V[i].tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=1.5, label=name, 
                   color=colors[i], alpha=0.7)
            ax.fill(angles, values, alpha=0.05, color=colors[i])

        v_plus_vals = self.V_plus.tolist()
        v_plus_vals += v_plus_vals[:1]
        ax.plot(angles, v_plus_vals, 'k--', linewidth=2, label='正理想解', 
               marker='*', markersize=10)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.criteria_names, fontsize=11)
        ax.set_title('TOPSIS 加权标准化雷达图', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return fig

    def plot_weights(self, figsize=(8, 5), save_path=None):
        """绘制权重饼图"""
        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.Set3(np.linspace(0, 1, len(self.criteria_names)))
        explode = [0.05] * len(self.criteria_names)

        wedges, texts, autotexts = ax.pie(
            self.weights, labels=self.criteria_names, autopct='%1.2f%%',
            colors=colors, explode=explode, shadow=True, startangle=90
        )

        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')

        ax.set_title('指标权重分布', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return fig

    def plot_full_table(self, figsize=(14, 8), save_path=None):
        """绘制综合评价结果表格"""
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')

        fig.text(0.5, 0.95, 'TOPSIS 多属性决策分析 — 综合评价结果表', 
                 ha='center', fontsize=18, fontweight='bold', color='#1a1a2e')

        table_data = []
        for rank, idx in enumerate(self.rankings, 1):
            score = self.scores[idx]
            stars = '★' * int(score * 5) + '☆' * (5 - int(score * 5))

            if score >= 0.7:
                level = '优秀'
            elif score >= 0.5:
                level = '良好'
            elif score >= 0.3:
                level = '一般'
            else:
                level = '较差'

            table_data.append([
                f'第{rank}名',
                self.alternative_names[idx],
                f'{self.D_plus[idx]:.4f}',
                f'{self.D_minus[idx]:.4f}',
                f'{score:.4f}',
                stars,
                level
            ])

        columns = ['排名', '方案', 'D+ (正理想解距离)', 'D- (负理想解距离)', 
                   '贴近度 C*', '星级评价', '等级']

        table = ax.table(
            cellText=table_data,
            colLabels=columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.08, 0.12, 0.18, 0.18, 0.14, 0.14, 0.10]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(13)
        table.scale(1.3, 2.8)

        for j in range(len(columns)):
            cell = table[(0, j)]
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold', fontsize=13)
            cell.set_height(0.08)

        for i in range(1, len(table_data) + 1):
            score_val = self.scores[self.rankings[i-1]]

            if score_val >= 0.7:
                bg_color = '#d5f5e3'
            elif score_val >= 0.5:
                bg_color = '#fcf3cf'
            elif score_val >= 0.3:
                bg_color = '#fadbd8'
            else:
                bg_color = '#f5b7b1'

            for j in range(len(columns)):
                cell = table[(i, j)]
                cell.set_facecolor(bg_color)
                cell.set_edgecolor('#34495e')
                cell.set_linewidth(1.5)

                if j == 0:
                    cell.set_text_props(fontweight='bold', fontsize=14)
                if j == 4:
                    cell.set_text_props(fontweight='bold', fontsize=14, color='#c0392b')
                if j == 6:
                    if score_val >= 0.7:
                        cell.set_text_props(fontweight='bold', color='#27ae60')
                    elif score_val >= 0.5:
                        cell.set_text_props(fontweight='bold', color='#f39c12')
                    elif score_val >= 0.3:
                        cell.set_text_props(fontweight='bold', color='#e67e22')
                    else:
                        cell.set_text_props(fontweight='bold', color='#e74c3c')

        fig.text(0.5, 0.08, 
            '注: D+ 越小越好 | D- 越大越好 | 贴近度 C* ∈ [0,1]，越接近1越优 | 权重由熵权法自动计算',
            ha='center', fontsize=11, style='italic', color='#7f8c8d')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        plt.show()
        return fig

    def export_excel(self, path):
        """导出结果到 Excel"""
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            self.df_result.to_excel(writer, sheet_name='评价结果', index=False)

            df_weights = pd.DataFrame({
                '指标': self.criteria_names,
                '权重': self.weights.round(4),
                '占比(%)': (self.weights * 100).round(2)
            })
            df_weights.to_excel(writer, sheet_name='权重', index=False)

            df_raw = pd.DataFrame(
                self.decision_matrix, 
                columns=self.criteria_names,
                index=self.alternative_names
            )
            df_raw.to_excel(writer, sheet_name='原始数据')

            df_R = pd.DataFrame(
                np.array(self.R),
                columns=self.criteria_names,
                index=self.alternative_names
            )
            df_R.to_excel(writer, sheet_name='标准化矩阵')

            df_V = pd.DataFrame(
                np.array(self.V),
                columns=self.criteria_names,
                index=self.alternative_names
            )
            df_V.to_excel(writer, sheet_name='加权标准化矩阵')

            df_ideal = pd.DataFrame({
                '指标': self.criteria_names,
                '正理想解 V+': self.V_plus.round(4),
                '负理想解 V-': self.V_minus.round(4)
            })
            df_ideal.to_excel(writer, sheet_name='理想解', index=False)

        print(f"结果已导出到: {path}")