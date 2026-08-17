import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ========== 中文字体设置 ==========
font_path = r'C:\Windows\Fonts\msyh.ttc'   # 微软雅黑

try:
    chinese_font = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = chinese_font.get_name()
except:
    # 如果找不到文件，尝试用字体名称
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    
plt.rcParams['axes.unicode_minus'] = False

class GM11:
    """
    GM(1,1) 灰色预测模型
    支持：建模、预测、精度检验、可视化
    """
    
    def __init__(self, data):
        """
        初始化模型
        
        Parameters:
        -----------
        data : array-like
            原始序列（非负数据），至少4个数据点
        """
        self.x0 = np.array(data, dtype=float).flatten()
        self.n = len(self.x0)
        
        if self.n < 4:
            raise ValueError("原始序列至少需要4个数据点")
        if np.any(self.x0 < 0):
            raise ValueError("原始序列必须为非负数")
            
        # 初始化各中间变量
        self.x1 = None           # 1-AGO累加序列
        self.z1 = None           # 紧邻均值生成序列
        self.a = None            # 发展系数
        self.b = None            # 灰色作用量
        self.x1_pred = None      # 累加序列预测值
        self.x0_pred = None      # 还原预测值
        self.residual = None     # 残差
        self.relative_error = None  # 相对误差
        self.C = None            # 后验差比
        self.P = None            # 小误差概率
        self.grade = None        # 精度等级
        
    def _build_matrix(self):
        """构建数据矩阵 B 和 Y"""
        # 1-AGO 累加生成
        self.x1 = np.cumsum(self.x0)
        
        # 紧邻均值生成序列 z1(k) = 0.5 * (x1(k) + x1(k-1))
        self.z1 = np.zeros(self.n - 1)
        for k in range(1, self.n):
            self.z1[k-1] = 0.5 * (self.x1[k] + self.x1[k-1])
        
        # 构建矩阵 B 和 Y
        B = np.column_stack((-self.z1, np.ones(self.n - 1)))
        Y = self.x0[1:].reshape(-1, 1)
        
        return B, Y
    
    def fit(self):
        """
        拟合 GM(1,1) 模型，求解参数 a, b
        
        Returns:
        --------
        self : 返回模型实例
        """
        B, Y = self._build_matrix()
        
        # 最小二乘法求解参数 [a, b]^T
        params = np.linalg.inv(B.T @ B) @ B.T @ Y
        self.a = float(params[0][0])
        self.b = float(params[1][0])
        
        # 计算累加序列的预测值
        self.x1_pred = np.zeros(self.n)
        for k in range(self.n):
            self.x1_pred[k] = (self.x0[0] - self.b / self.a) * np.exp(-self.a * k) + self.b / self.a
        
        # 还原预测值（累减）
        self.x0_pred = np.zeros(self.n)
        self.x0_pred[0] = self.x0[0]  # 第一个点保持不变
        for k in range(1, self.n):
            self.x0_pred[k] = self.x1_pred[k] - self.x1_pred[k-1]
        
        # 计算残差和相对误差
        self.residual = self.x0 - self.x0_pred
        self.relative_error = np.abs(self.residual / self.x0) * 100
        
        # 精度检验
        self._accuracy_test()
        
        return self
    
    def _accuracy_test(self):
        """精度检验：计算 C, P 和精度等级"""
        # 原始序列标准差 S1
        S1 = np.std(self.x0, ddof=1)
        
        # 残差标准差 S2
        S2 = np.std(self.residual, ddof=1)
        
        # 后验差比 C
        self.C = S2 / S1 if S1 != 0 else float('inf')
        
        # 残差均值
        mean_residual = np.mean(self.residual)
        
        # 小误差概率 P
        threshold = 0.6745 * S1
        count = np.sum(np.abs(self.residual - mean_residual) < threshold)
        self.P = count / self.n
        
        # 判断精度等级
        if self.C < 0.35 and self.P > 0.95:
            self.grade = "一级（优）"
        elif self.C < 0.50 and self.P > 0.80:
            self.grade = "二级（良）"
        elif self.C < 0.65 and self.P > 0.70:
            self.grade = "三级（合格）"
        else:
            self.grade = "四级（不合格）"
    
    def predict(self, steps=1):
        """
        预测未来 steps 个值
        
        Parameters:
        -----------
        steps : int
            预测步数
            
        Returns:
        --------
        predictions : ndarray
            预测值数组
        """
        if self.a is None:
            raise RuntimeError("请先调用 fit() 方法拟合模型")
        
        predictions = np.zeros(steps)
        for i in range(steps):
            k = self.n + i  # 从当前长度开始预测
            # 累加序列预测值
            x1_next = (self.x0[0] - self.b / self.a) * np.exp(-self.a * k) + self.b / self.a
            x1_prev = (self.x0[0] - self.b / self.a) * np.exp(-self.a * (k - 1)) + self.b / self.a
            predictions[i] = x1_next - x1_prev
        
        return predictions
    
    def summary(self):
        """打印模型摘要信息"""
        print("=" * 60)
        print("                    GM(1,1) 模型结果")
        print("=" * 60)
        print(f"\n原始序列: {self.x0}")
        print(f"\n模型参数:")
        print(f"   发展系数 a = {self.a:.6f}")
        print(f"   灰色作用量 b = {self.b:.6f}")
        print(f"\n时间响应函数:")
        print(f"   x^(1)(k+1) = ({self.x0[0]:.4f} - {self.b/self.a:.4f}) * e^(-{self.a:.4f}k) + {self.b/self.a:.4f}")
        print(f"\n拟合结果对比:")
        print(f"{'序号':>4} {'原始值':>10} {'预测值':>10} {'残差':>10} {'相对误差(%)':>12}")
        print("-" * 55)
        for i in range(self.n):
            print(f"{i+1:>4} {self.x0[i]:>10.4f} {self.x0_pred[i]:>10.4f} "
                  f"{self.residual[i]:>10.4f} {self.relative_error[i]:>12.2f}")
        
        print(f"\n精度检验:")
        print(f"   后验差比 C = {self.C:.4f}")
        print(f"   小误差概率 P = {self.P:.4f}")
        print(f"   平均相对误差 = {np.mean(self.relative_error):.2f}%")
        print(f"   精度等级: {self.grade}")
        print("=" * 60)
    
    def plot(self, future_steps=3, save_path=None):
        """
        可视化模型结果
        
        Parameters:
        -----------
        future_steps : int
            预测未来步数
        save_path : str, optional
            保存图片路径
        """
        if self.a is None:
            raise RuntimeError("请先调用 fit() 方法拟合模型")
        
        # 获取未来预测值
        future_pred = self.predict(future_steps)
        
        # 构建完整的时间轴
        t_original = np.arange(1, self.n + 1)
        t_future = np.arange(self.n + 1, self.n + future_steps + 1)
        t_all = np.arange(1, self.n + future_steps + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # ===== 图1: 原始数据 vs 拟合/预测数据 =====
        ax1 = axes[0, 0]
        ax1.plot(t_original, self.x0, 'bo-', linewidth=2, markersize=8, label='原始数据', zorder=5)
        ax1.plot(t_original, self.x0_pred, 'rs--', linewidth=2, markersize=6, label='拟合值', zorder=4)
        ax1.plot(t_future, future_pred, 'g^--', linewidth=2, markersize=8, label='预测值', zorder=4)
        ax1.axvline(x=self.n + 0.5, color='gray', linestyle=':', alpha=0.7, label='预测分界线')
        ax1.set_xlabel('时间序号', fontsize=12)
        ax1.set_ylabel('数值', fontsize=12)
        ax1.set_title('GM(1,1) 原始数据 vs 拟合/预测', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 添加数据标签
        for i, (x, y) in enumerate(zip(t_original, self.x0)):
            ax1.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontsize=9, color='blue')
        for i, (x, y) in enumerate(zip(t_future, future_pred)):
            ax1.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontsize=9, color='green')
        
        # ===== 图2: 1-AGO 累加序列 =====
        ax2 = axes[0, 1]
        ax2.plot(t_original, self.x1, 'bo-', linewidth=2, markersize=8, label='1-AGO原始', zorder=5)
        
        # 计算完整的累加预测序列
        x1_pred_all = np.zeros(self.n + future_steps)
        for k in range(self.n + future_steps):
            x1_pred_all[k] = (self.x0[0] - self.b / self.a) * np.exp(-self.a * k) + self.b / self.a
        
        ax2.plot(t_all, x1_pred_all, 'r--', linewidth=2, markersize=6, label='1-AGO拟合/预测', zorder=4)
        ax2.axvline(x=self.n + 0.5, color='gray', linestyle=':', alpha=0.7)
        ax2.set_xlabel('时间序号', fontsize=12)
        ax2.set_ylabel('累加值', fontsize=12)
        ax2.set_title('1-AGO 累加序列', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # ===== 图3: 残差分析 =====
        ax3 = axes[1, 0]
        colors = ['green' if e < 5 else 'orange' if e < 10 else 'red' for e in self.relative_error]
        bars = ax3.bar(t_original, self.residual, color=colors, edgecolor='black', alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.set_xlabel('时间序号', fontsize=12)
        ax3.set_ylabel('残差', fontsize=12)
        ax3.set_title('残差分析', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 添加残差数值标签
        for bar, val in zip(bars, self.residual):
            height = bar.get_height()
            ax3.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3 if height >= 0 else -15), textcoords="offset points",
                        ha='center', fontsize=9)
        
        # ===== 图4: 精度检验指标 =====
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # 构建信息文本
        info_text = (
            f"【模型参数】\n"
            f"  发展系数 a = {self.a:.6f}\n"
            f"  灰色作用量 b = {self.b:.6f}\n\n"
            f"【精度检验】\n"
            f"  后验差比 C = {self.C:.4f}\n"
            f"  小误差概率 P = {self.P:.4f}\n"
            f"  平均相对误差 = {np.mean(self.relative_error):.2f}%\n\n"
            f"【精度等级】\n"
            f"  {self.grade}\n\n"
            f"【预测结果】\n"
        )
        for i, val in enumerate(future_pred):
            info_text += f"  第 {self.n + i + 1} 期: {val:.4f}\n"
        
        ax4.text(0.1, 0.95, info_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, pad=1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        
        plt.show()
        
        return fig


# ==================== 使用示例 ====================

# 示例数据：某指标历年数值
data = [100, 115, 132, 155, 180, 210, 245, 290]

# 创建模型并拟合
model = GM11(data)
model.fit()

# 打印模型摘要
model.summary()

# 预测未来3期
future = model.predict(steps=3)
print(f"\n未来3期预测结果:")
for i, val in enumerate(future):
    print(f"   第 {model.n + i + 1} 期: {val:.4f}")

# 可视化（如需中文正常显示，请提前设置中文字体）
# plt.rcParams['font.family'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK JP', 'sans-serif']
# plt.rcParams['axes.unicode_minus'] = False
fig = model.plot(future_steps=3, save_path='gm11_result.png')