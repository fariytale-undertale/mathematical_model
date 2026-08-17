import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
# ==================================


class AHPCalculator:
    """层次分析法（AHP）计算器

    封装了 AHP 判断矩阵的一致性检验、权重计算和可视化功能。
    """

    # 随机一致性指标表
    RI_TABLE = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 
                6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

    def __init__(self, matrix, criteria=None):
        """初始化 AHP 计算器

        Parameters
        ----------
        matrix : np.ndarray
            AHP 判断矩阵（方阵）
        criteria : list[str], optional
            准则名称列表，默认为 C1, C2, ...
        """
        self.matrix = np.array(matrix, dtype=float)
        self.n = self.matrix.shape[0]
        self.criteria = criteria if criteria is not None else [f'C{i+1}' for i in range(self.n)]
        self._result = None

    def calculate(self):
        """计算一致性指标和权重向量

        Returns
        -------
        dict
            包含 lambda_max, CI, RI, CR, weights, consistent 的字典
        """
        eigenvalues, eigenvectors = np.linalg.eig(self.matrix)
        lambda_max = np.max(eigenvalues.real)
        CI = (lambda_max - self.n) / (self.n - 1)
        RI = self.RI_TABLE.get(self.n, 1.49)
        CR = CI / RI
        idx = np.argmax(eigenvalues.real)
        weights = eigenvectors[:, idx].real
        weights = weights / np.sum(weights)

        self._result = {
            'lambda_max': lambda_max,
            'CI': CI,
            'RI': RI,
            'CR': CR,
            'weights': weights,
            'consistent': CR < 0.10
        }
        return self._result

    @property
    def result(self):
        """获取计算结果（懒计算）"""
        if self._result is None:
            self.calculate()
        return self._result

    def plot(self, figsize=(10, 12), dpi=150, save_path=None, show=True):
        """绘制 AHP 分析结果（三张子图）

        Parameters
        ----------
        figsize : tuple
            画布尺寸
        dpi : int
            分辨率
        save_path : str, optional
            保存路径，为 None 时不保存
        show : bool
            是否调用 plt.show()

        Returns
        -------
        matplotlib.figure.Figure
        """
        result = self.result

        fig, axes = plt.subplots(3, 1, figsize=figsize)

        # ---------- 子图1：原始判断矩阵 ----------
        ax1 = axes[0]
        ax1.axis('off')
        ax1.set_title('AHP 原始判断矩阵', fontsize=20, fontweight='bold', pad=20)

        matrix_str = [[f"{val:.2f}" for val in row] for row in self.matrix]
        criteria_padded = ['  ' + c + '  ' for c in self.criteria]

        table1 = ax1.table(
            cellText=matrix_str, rowLabels=criteria_padded, colLabels=self.criteria,
            loc='center', cellLoc='center', bbox=[0.0, 0.1, 0.9, 0.8]
        )
        table1.auto_set_font_size(False)
        table1.set_fontsize(14)
        table1.scale(1, 2)

        for i in range(self.n):
            table1[(0, i)].set_facecolor("#2B61A2")
            table1[(0, i)].set_text_props(color='white', fontweight='bold')
            table1[(i+1, -1)].set_facecolor("#7281A6")
            table1[(i+1, -1)].set_text_props(color='white', fontweight='bold')

        for i in range(self.n):
            table1[(i+1, i)].set_facecolor('#E7E6E6')

        # ---------- 子图2：关键数据指标 ----------
        ax2 = axes[1]
        ax2.axis('off')
        ax2.set_title('AHP 一致性检验关键指标', fontsize=20, fontweight='bold', pad=20)

        metrics_data = [
            ['最大特征值 λ_max', f"{result['lambda_max']:.4f}"],
            ['一致性指标 CI', f"{result['CI']:.4f}"],
            ['随机一致性指标 RI', f"{result['RI']}"],
            ['一致性比例 CR', f"{result['CR']:.4f}"],
            ['一致性判断', '通过' if result['consistent'] else '不通过']
        ]

        table2 = ax2.table(
            cellText=metrics_data, colLabels=['指标名称', '数值'],
            loc='center', cellLoc='center', bbox=[0.0, 0.1, 0.9, 0.8]
        )
        table2.auto_set_font_size(False)
        table2.set_fontsize(14)
        table2.scale(1, 2.2)

        for i in range(2):
            table2[(0, i)].set_facecolor("#7281A6")
            table2[(0, i)].set_text_props(color='white', fontweight='bold')

        color = '#C6EFCE' if result['consistent'] else '#FFC7CE'
        text_color = '#006100' if result['consistent'] else '#9C0006'
        table2[(5, 0)].set_facecolor(color)
        table2[(5, 1)].set_facecolor(color)
        table2[(5, 0)].set_text_props(color=text_color, fontweight='bold')
        table2[(5, 1)].set_text_props(color=text_color, fontweight='bold')

        for i in range(1, 5):
            if i % 2 == 0:
                for j in range(2):
                    table2[(i, j)].set_facecolor('#F2F2F2')

        # ---------- 子图3：权重向量 ----------
        ax3 = axes[2]
        ax3.axis('off')
        ax3.set_title('AHP 准则权重向量', fontsize=20, fontweight='bold', pad=20)

        weights_data = []
        for i, w in enumerate(result['weights']):
            weights_data.append([self.criteria[i], f'{w:.4f}', f'{w*100:.2f}%'])

        table3 = ax3.table(
            cellText=weights_data, colLabels=['准则', '权重值', '百分比'],
            loc='center', cellLoc='center', bbox=[0.0, 0.1, 0.9, 0.8]
        )
        table3.auto_set_font_size(False)
        table3.set_fontsize(14)
        table3.scale(1, 2.2)

        for i in range(3):
            table3[(0, i)].set_facecolor('#4472C4')
            table3[(0, i)].set_text_props(color='white', fontweight='bold')

        for i in range(1, len(result['weights'])+1):
            if i % 2 == 0:
                for j in range(3):
                    table3[(i, j)].set_facecolor('#F2F2F2')

        plt.tight_layout(pad=3.0)

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def get_weights(self):
        """获取权重向量"""
        return self.result['weights']

    def is_consistent(self):
        """判断是否通过一致性检验"""
        return self.result['consistent']
