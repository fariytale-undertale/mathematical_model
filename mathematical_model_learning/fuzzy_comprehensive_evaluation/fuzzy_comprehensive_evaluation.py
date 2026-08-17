
"""
模糊综合评价 (Fuzzy Comprehensive Evaluation) 模块
===============================================
提供完整的模糊综合评价实现，支持多种模糊算子、隶属度函数和权重确定方法。

Author:
Date: 2026-07-21
"""

import numpy as np
from typing import List, Callable, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings


# ============================================================================
# 一、隶属度函数 (Membership Functions)
# ============================================================================

class MembershipFunction:
    """隶属度函数基类"""

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class TriangularMF(MembershipFunction):
    """
    三角隶属度函数 (Triangular Membership Function)

    参数:
        a: 左端点 (隶属度=0)
        b: 顶点 (隶属度=1)
        c: 右端点 (隶属度=0)
    """

    def __init__(self, a: float, b: float, c: float):
        if not (a <= b <= c):
            raise ValueError("必须满足 a <= b <= c")
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x)

        left = (x >= self.a) & (x < self.b)
        right = (x >= self.b) & (x <= self.c)

        if self.b != self.a:
            result[left] = (x[left] - self.a) / (self.b - self.a)
        if self.c != self.b:
            result[right] = (self.c - x[right]) / (self.c - self.b)

        result[x == self.b] = 1.0

        return result if result.size > 1 else float(result.item())

    def __repr__(self):
        return f"TriangularMF(a={self.a}, b={self.b}, c={self.c})"


class TrapezoidalMF(MembershipFunction):
    """
    梯形隶属度函数 (Trapezoidal Membership Function)

    参数:
        a: 左端点 (隶属度=0)
        b: 左肩点 (隶属度=1)
        c: 右肩点 (隶属度=1)
        d: 右端点 (隶属度=0)
    """

    def __init__(self, a: float, b: float, c: float, d: float):
        if not (a <= b <= c <= d):
            raise ValueError("必须满足 a <= b <= c <= d")
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x)

        left = (x >= self.a) & (x < self.b)
        flat = (x >= self.b) & (x <= self.c)
        right = (x > self.c) & (x <= self.d)

        if self.b != self.a:
            result[left] = (x[left] - self.a) / (self.b - self.a)
        result[flat] = 1.0
        if self.d != self.c:
            result[right] = (self.d - x[right]) / (self.d - self.c)

        return result if result.size > 1 else float(result.item())

    def __repr__(self):
        return f"TrapezoidalMF(a={self.a}, b={self.b}, c={self.c}, d={self.d})"


class GaussianMF(MembershipFunction):
    """
    高斯隶属度函数 (Gaussian Membership Function)

    参数:
        center: 中心点
        sigma: 标准差
    """

    def __init__(self, center: float, sigma: float):
        if sigma <= 0:
            raise ValueError("sigma 必须大于 0")
        self.center = center
        self.sigma = sigma

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x = np.asarray(x, dtype=float)
        result = np.exp(-0.5 * ((x - self.center) / self.sigma) ** 2)
        return result if result.size > 1 else float(result.item())

    def __repr__(self):
        return f"GaussianMF(center={self.center}, sigma={self.sigma})"


class SigmoidMF(MembershipFunction):
    """
    Sigmoid 隶属度函数

    参数:
        a: 斜率控制参数
        c: 中心点
    """

    def __init__(self, a: float, c: float):
        self.a = a
        self.c = c

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x = np.asarray(x, dtype=float)
        result = 1.0 / (1.0 + np.exp(-self.a * (x - self.c)))
        return result if result.size > 1 else float(result.item())

    def __repr__(self):
        return f"SigmoidMF(a={self.a}, c={self.c})"


class ZShapedMF(MembershipFunction):
    """Z 型隶属度函数 (递减 S 型)"""

    def __init__(self, a: float, b: float):
        if a >= b:
            raise ValueError("必须满足 a < b")
        self.a = a
        self.b = b

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x, dtype=float)

        left = x <= self.a
        mid = (x > self.a) & (x < (self.a + self.b) / 2)
        right_mid = (x >= (self.a + self.b) / 2) & (x < self.b)
        right = x >= self.b

        result[left] = 1.0
        result[mid] = 1 - 2 * ((x[mid] - self.a) / (self.b - self.a)) ** 2
        result[right_mid] = 2 * ((x[right_mid] - self.b) / (self.b - self.a)) ** 2
        result[right] = 0.0

        return result if result.size > 1 else float(result.item())


class SShapedMF(MembershipFunction):
    """S 型隶属度函数 (递增 S 型)"""

    def __init__(self, a: float, b: float):
        if a >= b:
            raise ValueError("必须满足 a < b")
        self.a = a
        self.b = b

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x, dtype=float)

        left = x <= self.a
        mid = (x > self.a) & (x < (self.a + self.b) / 2)
        right_mid = (x >= (self.a + self.b) / 2) & (x < self.b)
        right = x >= self.b

        result[left] = 0.0
        result[mid] = 2 * ((x[mid] - self.a) / (self.b - self.a)) ** 2
        result[right_mid] = 1 - 2 * ((x[right_mid] - self.b) / (self.b - self.a)) ** 2
        result[right] = 1.0

        return result if result.size > 1 else float(result.item())


# ============================================================================
# 二、模糊算子 (Fuzzy Operators)
# ============================================================================

class FuzzyOperator(Enum):
    """模糊合成算子枚举"""
    M_MIN_MAX = "M(∧,∨)"          # 主因素决定型 (取小-取大)
    M_DOT_MAX = "M(·,∨)"          # 主因素突出型 (乘积-取大)
    M_MIN_PLUS = "M(∧,⊕)"         # 加权平均型 (取小-有界和)
    M_DOT_PLUS = "M(·,+)"         # 加权平均型 (乘积-普通和)
    M_MIN_MIN = "M(∧,∧)"          # 取小-取小
    M_DOT_MIN = "M(·,∧)"          # 乘积-取小
    M_DOT_DOT = "M(·,·)"          # 乘积-乘积


def fuzzy_and(a: np.ndarray, b: np.ndarray, method: str = "min") -> np.ndarray:
    """
    模糊与运算

    参数:
        a, b: 输入数组
        method: "min" 或 "prod"
    """
    if method == "min":
        return np.minimum(a, b)
    elif method == "prod":
        return a * b
    else:
        raise ValueError(f"不支持的模糊与运算: {method}")


def fuzzy_or(a: np.ndarray, b: np.ndarray, method: str = "max") -> np.ndarray:
    """
    模糊或运算

    参数:
        a, b: 输入数组
        method: "max" 或 "sum" 或 "bounded_sum"
    """
    if method == "max":
        return np.maximum(a, b)
    elif method == "sum":
        return a + b
    elif method == "bounded_sum":
        return np.minimum(a + b, 1.0)
    else:
        raise ValueError(f"不支持的模糊或运算: {method}")


# ============================================================================
# 三、权重确定方法 (Weight Determination Methods)
# ============================================================================

class WeightMethod:
    """权重确定方法基类"""

    @staticmethod
    def entropy_weight(data: np.ndarray) -> np.ndarray:
        """
        熵权法确定权重

        参数:
            data: 原始数据矩阵 (n_samples, n_features)
        返回:
            权重向量 (n_features,)
        """
        # 数据归一化 (正向化)
        data = np.asarray(data, dtype=float)
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)

        # 避免除零
        ranges = data_max - data_min
        ranges[ranges == 0] = 1e-10

        normalized = (data - data_min) / ranges
        normalized = np.clip(normalized, 1e-10, 1.0)

        # 计算熵值
        n = data.shape[0]
        p = normalized / normalized.sum(axis=0, keepdims=True)
        p = np.clip(p, 1e-10, 1.0)

        entropy = -np.sum(p * np.log(p), axis=0) / np.log(n)

        # 计算权重
        redundancy = 1 - entropy
        weights = redundancy / redundancy.sum()

        return weights

    @staticmethod
    def ahp_weight(pairwise_matrix: np.ndarray, 
                   max_iter: int = 100, 
                   tol: float = 1e-6) -> Tuple[np.ndarray, float, float]:
        """
        AHP 层次分析法确定权重

        参数:
            pairwise_matrix: 判断矩阵 (n, n)
            max_iter: 最大迭代次数
            tol: 收敛容差
        返回:
            (权重向量, 最大特征值, 一致性比率 CR)
        """
        n = pairwise_matrix.shape[0]

        # 幂法求最大特征值和特征向量
        w = np.ones(n) / n
        for _ in range(max_iter):
            w_new = pairwise_matrix @ w
            w_new = w_new / w_new.sum()
            if np.linalg.norm(w_new - w) < tol:
                break
            w = w_new

        # 计算最大特征值
        lambda_max = np.sum((pairwise_matrix @ w) / w) / n

        # 一致性检验
        CI = (lambda_max - n) / (n - 1)
        RI_table = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 
                    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        RI = RI_table.get(n, 1.49)
        CR = CI / RI if RI > 0 else 0

        return w, lambda_max, CR

    @staticmethod
    def variance_weight(data: np.ndarray) -> np.ndarray:
        """
        变异系数法确定权重

        参数:
            data: 原始数据矩阵 (n_samples, n_features)
        返回:
            权重向量 (n_features,)
        """
        data = np.asarray(data, dtype=float)
        mean = data.mean(axis=0)
        std = data.std(axis=0, ddof=1)

        # 变异系数
        cv = std / (np.abs(mean) + 1e-10)
        weights = cv / cv.sum()

        return weights

    @staticmethod
    def standard_weight(data: np.ndarray) -> np.ndarray:
        """
        标准离差法确定权重

        参数:
            data: 原始数据矩阵 (n_samples, n_features)
        返回:
            权重向量 (n_features,)
        """
        data = np.asarray(data, dtype=float)
        std = data.std(axis=0, ddof=1)
        weights = std / std.sum()

        return weights


# ============================================================================
# 四、模糊综合评价核心类
# ============================================================================

@dataclass
class FCEConfig:
    """模糊综合评价配置"""
    operator: FuzzyOperator = FuzzyOperator.M_DOT_PLUS
    normalize_weights: bool = True
    verbose: bool = False


class FuzzyComprehensiveEvaluation:
    """
    模糊综合评价类 (Fuzzy Comprehensive Evaluation)

    支持单级和多级模糊综合评价，提供多种模糊算子和权重确定方法。

    使用方法:
        1. 初始化评价对象
        2. 设置因素集、评语集、权重向量
        3. 构建模糊关系矩阵
        4. 调用 evaluate() 进行评价

    示例:
        >>> fce = FuzzyComprehensiveEvaluation()
        >>> fce.set_factors(["质量", "价格", "服务"])
        >>> fce.set_comments(["优", "良", "中", "差"])
        >>> fce.set_weights([0.4, 0.3, 0.3])
        >>> fce.set_relation_matrix(R)
        >>> result = fce.evaluate()
    """

    def __init__(self, config: Optional[FCEConfig] = None):
        self.config = config or FCEConfig()

        # 因素集 (评价指标)
        self.factors: Optional[List[str]] = None
        self.n_factors: int = 0

        # 评语集 (评价等级)
        self.comments: Optional[List[str]] = None
        self.n_comments: int = 0

        # 权重向量
        self.weights: Optional[np.ndarray] = None

        # 模糊关系矩阵 R (n_factors × n_comments)
        self.relation_matrix: Optional[np.ndarray] = None

        # 多级评价子系统
        self.sub_systems: Optional[List['FuzzyComprehensiveEvaluation']] = None
        self.sub_factor_indices: Optional[List[List[int]]] = None

        # 评价结果
        self.result: Optional[np.ndarray] = None
        self.score: Optional[float] = None

    # ------------------------------------------------------------------------
    # 设置方法
    # ------------------------------------------------------------------------

    def set_factors(self, factors: List[str]) -> 'FuzzyComprehensiveEvaluation':
        """设置因素集"""
        self.factors = list(factors)
        self.n_factors = len(factors)
        return self

    def set_comments(self, comments: List[str]) -> 'FuzzyComprehensiveEvaluation':
        """设置评语集"""
        self.comments = list(comments)
        self.n_comments = len(comments)
        return self

    def set_weights(self, weights: Union[List[float], np.ndarray]) -> 'FuzzyComprehensiveEvaluation':
        """
        设置权重向量

        参数:
            weights: 权重列表或数组，长度必须等于因素数
        """
        self.weights = np.asarray(weights, dtype=float)

        if self.config.normalize_weights:
            self.weights = self.weights / self.weights.sum()

        return self

    def auto_weights(self, data: np.ndarray, method: str = "entropy") -> 'FuzzyComprehensiveEvaluation':
        """
        自动确定权重

        参数:
            data: 原始数据矩阵
            method: "entropy" | "variance" | "standard"
        """
        if method == "entropy":
            self.weights = WeightMethod.entropy_weight(data)
        elif method == "variance":
            self.weights = WeightMethod.variance_weight(data)
        elif method == "standard":
            self.weights = WeightMethod.standard_weight(data)
        else:
            raise ValueError(f"不支持的权重确定方法: {method}")

        if self.config.verbose:
            print(f"自动权重 ({method}): {self.weights}")

        return self

    def set_relation_matrix(self, R: Union[List[List[float]], np.ndarray]) -> 'FuzzyComprehensiveEvaluation':
        """
        设置模糊关系矩阵

        参数:
            R: 模糊关系矩阵，形状为 (n_factors, n_comments)
               R[i][j] 表示第 i 个因素对第 j 个评语的隶属度
        """
        self.relation_matrix = np.asarray(R, dtype=float)

        # 验证矩阵形状
        if self.n_factors > 0 and self.relation_matrix.shape[0] != self.n_factors:
            raise ValueError(
                f"关系矩阵行数 ({self.relation_matrix.shape[0]}) "
                f"与因素数 ({self.n_factors}) 不匹配"
            )
        if self.n_comments > 0 and self.relation_matrix.shape[1] != self.n_comments:
            raise ValueError(
                f"关系矩阵列数 ({self.relation_matrix.shape[1]}) "
                f"与评语数 ({self.n_comments}) 不匹配"
            )

        # 自动更新维度
        if self.n_factors == 0:
            self.n_factors = self.relation_matrix.shape[0]
        if self.n_comments == 0:
            self.n_comments = self.relation_matrix.shape[1]

        return self

    def build_relation_from_membership(self, 
                                        values: np.ndarray,
                                        membership_functions: List[List[MembershipFunction]]) -> 'FuzzyComprehensiveEvaluation':
        """
        从隶属度函数构建模糊关系矩阵

        参数:
            values: 各因素的实际值 (n_factors,)
            membership_functions: 每个因素对应的隶属度函数列表
                                   形状为 (n_factors, n_comments)
        """
        values = np.asarray(values, dtype=float)
        n_factors = len(membership_functions)
        n_comments = len(membership_functions[0])

        R = np.zeros((n_factors, n_comments))
        for i in range(n_factors):
            for j in range(n_comments):
                R[i, j] = membership_functions[i][j](values[i])

        self.set_relation_matrix(R)
        return self

    def build_relation_from_survey(self, 
                                    survey_data: np.ndarray) -> 'FuzzyComprehensiveEvaluation':
        """
        从调查统计数据构建模糊关系矩阵

        参数:
            survey_data: 调查统计矩阵 (n_factors, n_comments)
                         每个元素表示选择该评语的人数或频率
        """
        survey_data = np.asarray(survey_data, dtype=float)

        # 归一化，使每行和为 1
        row_sums = survey_data.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除零
        R = survey_data / row_sums

        self.set_relation_matrix(R)
        return self

    # ------------------------------------------------------------------------
    # 多级评价设置
    # ------------------------------------------------------------------------

    def set_sub_systems(self, 
                        sub_systems: List['FuzzyComprehensiveEvaluation'],
                        sub_factor_indices: List[List[int]]) -> 'FuzzyComprehensiveEvaluation':
        """
        设置多级评价的子系统

        参数:
            sub_systems: 子评价系统列表
            sub_factor_indices: 每个子系统对应的父级因素索引
        """
        self.sub_systems = sub_systems
        self.sub_factor_indices = sub_factor_indices
        return self

    # ------------------------------------------------------------------------
    # 核心评价方法
    # ------------------------------------------------------------------------

    def evaluate(self, 
                 operator: Optional[FuzzyOperator] = None) -> np.ndarray:
        """
        执行模糊综合评价

        参数:
            operator: 模糊合成算子，None 则使用配置中的默认算子
        返回:
            综合评价结果向量 (n_comments,)
        """
        op = operator or self.config.operator

        # 多级评价
        if self.sub_systems is not None:
            return self._evaluate_multi_level(op)

        # 单级评价
        return self._evaluate_single_level(op)

    def _evaluate_single_level(self, operator: FuzzyOperator) -> np.ndarray:
        """单级模糊综合评价"""
        self._validate_single_level()

        W = self.weights.reshape(1, -1)  # (1, n_factors)
        R = self.relation_matrix          # (n_factors, n_comments)

        # 根据算子类型选择合成方法
        if operator == FuzzyOperator.M_MIN_MAX:
            # M(∧,∨): 取小-取大
            B = np.max(np.minimum(W.T, R), axis=0)

        elif operator == FuzzyOperator.M_DOT_MAX:
            # M(·,∨): 乘积-取大
            B = np.max(W.T * R, axis=0)

        elif operator == FuzzyOperator.M_MIN_PLUS:
            # M(∧,⊕): 取小-有界和
            temp = np.minimum(W.T, R)
            B = np.minimum(temp.sum(axis=0), 1.0)

        elif operator == FuzzyOperator.M_DOT_PLUS:
            # M(·,+): 乘积-普通和 (加权平均型，最常用)
            B = (W @ R).flatten()

        elif operator == FuzzyOperator.M_MIN_MIN:
            # M(∧,∧): 取小-取小
            B = np.min(np.minimum(W.T, R), axis=0)

        elif operator == FuzzyOperator.M_DOT_MIN:
            # M(·,∧): 乘积-取小
            B = np.min(W.T * R, axis=0)

        elif operator == FuzzyOperator.M_DOT_DOT:
            # M(·,·): 乘积-乘积
            B = np.prod(W.T * R, axis=0)

        else:
            raise ValueError(f"不支持的模糊算子: {operator}")

        # 归一化结果
        if B.sum() > 0:
            B = B / B.sum()

        self.result = B

        if self.config.verbose:
            print(f"使用算子: {operator.value}")
            print(f"评价结果: {B}")

        return B

    def _evaluate_multi_level(self, operator: FuzzyOperator) -> np.ndarray:
        """多级模糊综合评价"""
        if self.sub_systems is None or self.sub_factor_indices is None:
            raise ValueError("多级评价需要设置子系统")

        # 先计算各子系统的评价结果
        sub_results = []
        for sub_sys in self.sub_systems:
            sub_result = sub_sys.evaluate(operator)
            sub_results.append(sub_result)

        # 构建新的模糊关系矩阵 (子系统数 × 评语数)
        R_multi = np.vstack(sub_results)

        # 使用父级权重进行合成
        old_R = self.relation_matrix
        self.relation_matrix = R_multi

        result = self._evaluate_single_level(operator)

        # 恢复原始关系矩阵
        self.relation_matrix = old_R

        return result

    def _validate_single_level(self):
        """验证单级评价所需数据是否完整"""
        if self.weights is None:
            raise ValueError("权重向量未设置")
        if self.relation_matrix is None:
            raise ValueError("模糊关系矩阵未设置")
        if self.n_factors == 0:
            raise ValueError("因素集未设置")
        if self.n_comments == 0:
            raise ValueError("评语集未设置")

        if len(self.weights) != self.n_factors:
            raise ValueError(
                f"权重维度 ({len(self.weights)}) 与因素数 ({self.n_factors}) 不匹配"
            )
        if self.relation_matrix.shape != (self.n_factors, self.n_comments):
            raise ValueError(
                f"关系矩阵形状 {self.relation_matrix.shape} 与 "
                f"(因素数={self.n_factors}, 评语数={self.n_comments}) 不匹配"
            )

    # ------------------------------------------------------------------------
    # 结果分析
    # ------------------------------------------------------------------------

    def compute_score(self, comment_scores: Optional[Union[List[float], np.ndarray]] = None) -> float:
        """
        计算综合得分

        参数:
            comment_scores: 各评语对应的分值，None 则使用默认 1~n
        返回:
            综合得分
        """
        if self.result is None:
            raise ValueError("请先调用 evaluate() 进行评价")

        if comment_scores is None:
            scores = np.arange(self.n_comments, 0, -1)  # 递减，如 [4,3,2,1]
        else:
            scores = np.asarray(comment_scores, dtype=float)

        if len(scores) != self.n_comments:
            raise ValueError("评语分值数量与评语数不匹配")

        self.score = float(np.dot(self.result, scores))
        return self.score

    def get_max_membership_comment(self) -> Tuple[str, float]:
        """
        最大隶属度原则确定评语等级

        返回:
            (评语, 隶属度)
        """
        if self.result is None:
            raise ValueError("请先调用 evaluate() 进行评价")

        max_idx = np.argmax(self.result)
        comment = self.comments[max_idx] if self.comments else f"等级{max_idx + 1}"
        return comment, float(self.result[max_idx])

    def get_weighted_comment(self, comment_scores: Optional[Union[List[float], np.ndarray]] = None) -> Tuple[str, float]:
        """
        加权平均原则确定评语等级

        返回:
            (评语, 综合得分)
        """
        score = self.compute_score(comment_scores)

        # 找到最接近的评语
        if comment_scores is None:
            comment_scores = np.arange(self.n_comments, 0, -1)
        else:
            comment_scores = np.asarray(comment_scores, dtype=float)

        closest_idx = np.argmin(np.abs(comment_scores - score))
        comment = self.comments[closest_idx] if self.comments else f"等级{closest_idx + 1}"

        return comment, score

    def sensitivity_analysis(self, 
                            perturbation_range: float = 0.2,
                            n_samples: int = 100) -> dict:
        """
        敏感性分析：扰动权重，观察结果变化

        参数:
            perturbation_range: 扰动范围
            n_samples: 采样次数
        返回:
            分析结果字典
        """
        if self.weights is None or self.relation_matrix is None:
            raise ValueError("权重和关系矩阵必须已设置")

        original_weights = self.weights.copy()
        original_result = self.evaluate().copy()

        results = []
        for _ in range(n_samples):
            # 随机扰动权重
            noise = np.random.uniform(-perturbation_range, perturbation_range, size=self.n_factors)
            perturbed = np.clip(original_weights + noise, 0.01, 1.0)
            perturbed = perturbed / perturbed.sum()

            self.weights = perturbed
            result = self.evaluate()
            results.append(result)

        # 恢复原始权重
        self.weights = original_weights
        self.result = original_result

        results = np.array(results)

        return {
            "mean": results.mean(axis=0),
            "std": results.std(axis=0),
            "min": results.min(axis=0),
            "max": results.max(axis=0),
            "original": original_result
        }

    # ------------------------------------------------------------------------
    # 可视化与输出
    # ------------------------------------------------------------------------

    def summary(self) -> str:
        """生成评价结果摘要"""
        if self.result is None:
            return "尚未进行评价"

        lines = ["=" * 50]
        lines.append("模糊综合评价结果")
        lines.append("=" * 50)

        if self.factors:
            lines.append(f"因素集: {self.factors}")
        if self.comments:
            lines.append(f"评语集: {self.comments}")
        if self.weights is not None:
            lines.append(f"权重: {self.weights.round(4).tolist()}")

        lines.append("-" * 50)
        lines.append("评价结果向量:")

        for i, val in enumerate(self.result):
            label = self.comments[i] if self.comments else f"等级{i+1}"
            lines.append(f"  {label}: {val:.4f}")

        lines.append("-" * 50)

        max_comment, max_val = self.get_max_membership_comment()
        lines.append(f"最大隶属度原则: {max_comment} (隶属度={max_val:.4f})")

        if self.score is not None:
            lines.append(f"综合得分: {self.score:.4f}")

        lines.append("=" * 50)

        return "\n".join(lines)

    def __repr__(self):
        return f"FuzzyComprehensiveEvaluation(factors={self.n_factors}, comments={self.n_comments})"


# ============================================================================
# 五、实用工具函数
# ============================================================================

def build_ahp_matrix(comparisons: List[Tuple[int, int, float]]) -> np.ndarray:
    """
    从成对比较构建 AHP 判断矩阵

    参数:
        comparisons: 比较元组列表 [(i, j, value), ...]
                     表示因素 i 相对于因素 j 的重要性为 value
    返回:
        判断矩阵
    """
    n = max(max(i, j) for i, j, _ in comparisons) + 1
    matrix = np.ones((n, n))

    for i, j, val in comparisons:
        matrix[i, j] = val
        matrix[j, i] = 1.0 / val

    return matrix


def normalize_matrix(data: np.ndarray, 
                     direction: str = "forward") -> np.ndarray:
    """
    矩阵归一化

    参数:
        data: 原始数据矩阵
        direction: "forward" (正向指标，越大越好) 或 "reverse" (逆向指标，越小越好)
    """
    data = np.asarray(data, dtype=float)
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)

    ranges = data_max - data_min
    ranges[ranges == 0] = 1e-10

    if direction == "forward":
        return (data - data_min) / ranges
    elif direction == "reverse":
        return (data_max - data) / ranges
    else:
        raise ValueError(f"不支持的归一化方向: {direction}")


# ============================================================================
# 六、完整示例
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("模糊综合评价示例")
    print("=" * 60)

    # 示例 1: 单级模糊综合评价 (产品质量评价)
    print("\n【示例 1】单级模糊综合评价 —— 产品质量评价")
    print("-" * 60)

    fce = FuzzyComprehensiveEvaluation(config=FCEConfig(
        operator=FuzzyOperator.M_DOT_PLUS,
        verbose=True
    ))

    # 设置因素集和评语集
    fce.set_factors(["外观", "性能", "耐用性", "价格"])
    fce.set_comments(["优秀", "良好", "一般", "较差"])

    # 设置权重
    fce.set_weights([0.25, 0.35, 0.25, 0.15])

    # 设置模糊关系矩阵 (来自专家打分或调查统计)
    # 行: 因素, 列: 评语
    R = np.array([
        [0.3, 0.4, 0.2, 0.1],   # 外观
        [0.5, 0.3, 0.15, 0.05], # 性能
        [0.2, 0.4, 0.3, 0.1],   # 耐用性
        [0.1, 0.3, 0.4, 0.2]    # 价格
    ])
    fce.set_relation_matrix(R)

    # 执行评价
    result = fce.evaluate()

    # 计算综合得分 (优秀=4, 良好=3, 一般=2, 较差=1)
    score = fce.compute_score([4, 3, 2, 1])

    print(f"\n评价结果向量: {result.round(4)}")
    print(f"综合得分: {score:.4f}")

    max_comment, max_val = fce.get_max_membership_comment()
    print(f"最大隶属度评语: {max_comment} (隶属度={max_val:.4f})")

    print(f"\n{fce.summary()}")

    # 示例 2: 使用隶属度函数构建关系矩阵
    print("\n【示例 2】基于隶属度函数的评价 —— 水质评价")
    print("-" * 60)

    fce2 = FuzzyComprehensiveEvaluation()
    fce2.set_factors(["溶解氧", "氨氮", "总磷", "COD"])
    fce2.set_comments(["I类", "II类", "III类", "IV类", "V类"])
    fce2.set_weights([0.3, 0.25, 0.25, 0.2])

    # 实际测量值
    measured_values = np.array([7.5, 0.8, 0.15, 15.0])

    # 为每个因素定义各等级的隶属度函数
    # 简化示例：使用三角/梯形函数
    mfs = [
        # 溶解氧 (mg/L): 越高越好
        [TrapezoidalMF(7, 7.5, 100, 100),   # I类
         TrapezoidalMF(6, 6.5, 7.5, 8),     # II类
         TrapezoidalMF(5, 5.5, 6.5, 7),     # III类
         TrapezoidalMF(3, 3.5, 5.5, 6),     # IV类
         TrapezoidalMF(0, 0, 3.5, 4)],      # V类

        # 氨氮 (mg/L): 越低越好
        [TrapezoidalMF(0, 0, 0.15, 0.2),    # I类
         TrapezoidalMF(0.15, 0.2, 0.5, 0.6),# II类
         TrapezoidalMF(0.5, 0.6, 1.0, 1.2), # III类
         TrapezoidalMF(1.0, 1.2, 1.5, 2.0), # IV类
         TrapezoidalMF(1.5, 2.0, 100, 100)],# V类

        # 总磷 (mg/L)
        [TrapezoidalMF(0, 0, 0.02, 0.03),
         TrapezoidalMF(0.02, 0.03, 0.05, 0.06),
         TrapezoidalMF(0.05, 0.06, 0.1, 0.12),
         TrapezoidalMF(0.1, 0.12, 0.2, 0.25),
         TrapezoidalMF(0.2, 0.25, 100, 100)],

        # COD (mg/L)
        [TrapezoidalMF(0, 0, 15, 18),
         TrapezoidalMF(15, 18, 20, 22),
         TrapezoidalMF(20, 22, 25, 28),
         TrapezoidalMF(25, 28, 35, 40),
         TrapezoidalMF(35, 40, 100, 100)]
    ]

    fce2.build_relation_from_membership(measured_values, mfs)
    result2 = fce2.evaluate()
    score2 = fce2.compute_score([5, 4, 3, 2, 1])

    print(f"测量值: {measured_values}")
    print(f"模糊关系矩阵:\n{fce2.relation_matrix.round(4)}")
    print(f"评价结果: {result2.round(4)}")
    print(f"综合得分: {score2:.4f}")
    print(f"水质等级: {fce2.get_max_membership_comment()[0]}")

    # 示例 3: 熵权法自动确定权重
    print("\n【示例 3】熵权法自动确定权重")
    print("-" * 60)

    # 模拟多组样本数据
    sample_data = np.array([
        [85, 90, 78, 88],
        [75, 85, 82, 80],
        [90, 88, 85, 92],
        [70, 75, 80, 78],
        [88, 92, 90, 85]
    ])

    fce3 = FuzzyComprehensiveEvaluation()
    fce3.set_factors(["指标A", "指标B", "指标C", "指标D"])
    fce3.set_comments(["优", "良", "中", "差"])
    fce3.auto_weights(sample_data, method="entropy")

    # 构建关系矩阵 (对最后一组样本)
    R3 = np.array([
        [0.6, 0.3, 0.1, 0.0],
        [0.7, 0.2, 0.1, 0.0],
        [0.5, 0.4, 0.1, 0.0],
        [0.4, 0.4, 0.15, 0.05]
    ])
    fce3.set_relation_matrix(R3)

    result3 = fce3.evaluate()
    print(f"熵权法权重: {fce3.weights.round(4)}")
    print(f"评价结果: {result3.round(4)}")

    # 示例 4: 敏感性分析
    print("\n【示例 4】敏感性分析")
    print("-" * 60)

    sens = fce.sensitivity_analysis(perturbation_range=0.1, n_samples=50)
    print("原始结果:", sens["original"].round(4))
    print("均值:", sens["mean"].round(4))
    print("标准差:", sens["std"].round(4))

    print("\n" + "=" * 60)
    print("所有示例运行完毕！")
    print("=" * 60)