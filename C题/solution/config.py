"""
2024 CUMCM Problem C: Crop Planting Strategy
Configuration and Constants
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
ATTACHMENT1 = os.path.join(DATA_DIR, '附件1.xlsx')
ATTACHMENT2 = os.path.join(DATA_DIR, '附件2.xlsx')
ATTACHMENT3_DIR = os.path.join(DATA_DIR, '附件3')
RESULT1_1 = os.path.join(ATTACHMENT3_DIR, 'result1_1.xlsx')
RESULT1_2 = os.path.join(ATTACHMENT3_DIR, 'result1_2.xlsx')
RESULT2 = os.path.join(ATTACHMENT3_DIR, 'result2.xlsx')

# Ensure output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Planning horizon
YEARS = list(range(2024, 2031))
BASE_YEAR = 2023

# Land type categories
LAND_TYPES = {
    '平旱地': 'dry_flat',
    '梯田': 'terrace',
    '山坡地': 'hillside',
    '水浇地': 'irrigated',
    '普通大棚': 'greenhouse',
    '智慧大棚': 'smart_greenhouse',
}

# Plot name prefixes by type
PLOT_PREFIXES = {
    'A': 'dry_flat',      # 平旱地 A1-A6
    'B': 'terrace',        # 梯田 B1-B14
    'C': 'hillside',       # 山坡地 C1-C6
    'D': 'irrigated',      # 水浇地 D1-D8
    'E': 'greenhouse',     # 普通大棚 E1-E16
    'F': 'smart_greenhouse', # 智慧大棚 F1-F4
}

# Season modes
SINGLE_SEASON = 'single'        # 单季
DOUBLE_SEASON = 'double'        # 两季
SEASON1 = '第一季'
SEASON2 = '第二季'

# Crop categories
GRAIN_LEGUMES = list(range(1, 6))      # 粮食豆类: 1-5 (黄豆,黑豆,红豆,绿豆,爬豆)
GRAIN_CROPS = list(range(6, 16))        # 粮食: 6-15 (小麦,玉米,谷子,高粱,黍子,荞麦,南瓜,红薯,莜麦,大麦)
RICE = 16                                # 水稻
VEGETABLE_LEGUMES = [17, 18, 19]        # 蔬菜豆类: 豇豆,刀豆,芸豆
VEGETABLES = list(range(20, 35))         # 蔬菜: 20-34
WINTER_VEGETABLES = [35, 36, 37]        # 大白菜,白萝卜,红萝卜 (仅水浇地第二季)
MUSHROOMS = [38, 39, 40, 41]            # 食用菌: 榆黄菇,香菇,白灵菇,羊肚菌

ALL_LEGUMES = GRAIN_LEGUMES + VEGETABLE_LEGUMES  # 所有豆类
ALL_CROPS = list(range(1, 42))
N_CROPS = 41

# Crop names
CROP_NAMES = {
    1: '黄豆', 2: '黑豆', 3: '红豆', 4: '绿豆', 5: '爬豆',
    6: '小麦', 7: '玉米', 8: '谷子', 9: '高粱', 10: '黍子',
    11: '荞麦', 12: '南瓜', 13: '红薯', 14: '莜麦', 15: '大麦',
    16: '水稻',
    17: '豇豆', 18: '刀豆', 19: '芸豆',
    20: '土豆', 21: '西红柿', 22: '茄子', 23: '菠菜', 24: '青椒',
    25: '菜花', 26: '包菜', 27: '油麦菜', 28: '小青菜', 29: '黄瓜',
    30: '生菜', 31: '辣椒', 32: '空心菜', 33: '黄心菜', 34: '芹菜',
    35: '大白菜', 36: '白萝卜', 37: '红萝卜',
    38: '榆黄菇', 39: '香菇', 40: '白灵菇', 41: '羊肚菌',
}

# Problem parameters
# Uncertainty ranges (Problem 2)
WHEAT_CORN_GROWTH_RANGE = (0.05, 0.10)     # 小麦/玉米年增长率
OTHER_SALES_VARIATION = 0.05                 # 其他作物销量年变化
YIELD_VARIATION = 0.10                       # 亩产量年变化
COST_GROWTH = 0.05                           # 成本年增长
VEGETABLE_PRICE_GROWTH = 0.05                # 蔬菜售价年增长
MUSHROOM_PRICE_DECLINE = (0.01, 0.05)       # 食用菌售价年下降
MOREL_DECLINE = 0.05                         # 羊肚菌售价年下降

# Soft constraints
MIN_AREA_PER_CROP_OPEN = 1.0                 # 露天最小种植面积(亩)
MIN_AREA_PER_CROP_GREENHOUSE = 0.15          # 大棚最小种植面积(亩)
MAX_PLOTS_PER_CROP = 5                       # 每种作物每季最多分散地块数
MIN_AREA_HALF_PLOT = True                    # 最小面积≥半块地（C201约束）

# ─── Natural Disaster Risk Parameters (华北山区) ───
# 参考文献: 刘蕾(2012)南信大寒潮统计; 何兰英(2016)兰大华北干旱研究
# 寒潮发生于非冬季(春夏秋), 影响对应季次作物
COLD_WAVE_PROB = 0.10                        # 寒潮年发生概率 (~1次/10年, 50年26次→P≈0.5, 取保守)
COLD_WAVE_SEASONS = ['第一季', '第二季']      # 寒潮影响季节(非冬季=第一季+第二季)
COLD_WAVE_DAMAGE = {
    'vegetable': 0.30,    # 蔬菜减产30%
    'mushroom': 0.35,     # 食用菌减产35%
    'grain': 0.25,        # 粮食作物减产25%
    'rice': 0.25,         # 水稻减产25%
    'legume': 0.25,       # 豆类减产25%
}
# 干旱主要发生于夏季, 影响对应季次作物
DROUGHT_PROB = 0.096                          # 干旱年发生概率 (~9.6% 据兰大2016)
DROUGHT_SEASONS = ['第一季', '第二季', '单季']
DROUGHT_DAMAGE = {
    'resistant': 0.10,    # 抗旱作物减产10%
    'non_resistant': 0.40, # 非抗旱蔬菜减产40%
    'other': 0.25,        # 其他作物减产25%
}
# 抗旱作物列表 (小麦,玉米,谷子,高粱,大麦,土豆,西红柿,茄子,黄瓜,辣椒)
DROUGHT_RESISTANT_CROPS = [6, 7, 8, 9, 15, 20, 21, 22, 29, 31]
# 非抗旱蔬菜 (=所有蔬菜 - 抗旱蔬菜中的蔬菜)
DROUGHT_NON_RESISTANT_CROPS = [c for c in list(range(17, 35)) if c not in [20, 21, 22, 29, 31]]

# DP parameters
LAGRANGE_MAX_ITER = 200
LAGRANGE_TOL = 1e-4
SUBLINEAR_RHO_START = 2.0
