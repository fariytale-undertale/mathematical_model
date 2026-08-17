import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt

# ========== 中文字体配置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 数据加载与预处理 ==========
df = pd.read_csv('data.csv', parse_dates=['date'], index_col='date')
y = df['value'].asfreq('MS')  # 设定频率（月度月初）
y = y.dropna()

# ========== 2. 可视化 ==========
y.plot(title='原始序列')
plt.show()

# ========== 3. 平稳性检验 ==========
def check_stationarity(series):
    result = adfuller(series)
    print(f'ADF: {result[0]:.4f}, p-value: {result[1]:.4f}')
    return result[1] < 0.05

is_stationary = check_stationarity(y)
d = 0
temp = y.copy()
while not is_stationary and d < 3:
    d += 1
    temp = temp.diff().dropna()
    is_stationary = check_stationarity(temp)
print(f'差分阶数 d = {d}')

# ========== 4. ACF/PACF 定阶 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(temp, ax=axes[0], lags=20, title='ACF (差分后)')
plot_pacf(temp, ax=axes[1], lags=20, title='PACF (差分后)')
plt.show()

# ========== 5. 模型拟合与选择 ==========
best_aic = np.inf
best_order = None
best_model = None

for p in range(4):
    for q in range(4):
        try:
            model = ARIMA(y, order=(p, d, q))
            fitted = model.fit()
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = (p, d, q)
                best_model = fitted
        except:
            continue

print(f'最优模型: ARIMA{best_order}, AIC={best_aic:.2f}')
print(best_model.summary())

# ========== 6. 残差诊断 ==========
resid = best_model.resid

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
resid.plot(ax=axes[0,0], title='残差时序图')
resid.plot(kind='kde', ax=axes[0,1], title='残差密度')
plot_acf(resid, ax=axes[1,0], lags=20, title='残差ACF')
axes[1,1].hist(resid, bins=20, edgecolor='black')
axes[1,1].set_title('残差直方图')
plt.tight_layout()
plt.show()

# Ljung-Box 检验
lb = acorr_ljungbox(resid, lags=10, return_df=True)
print(lb)

# ========== 7. 预测与可视化 ==========
forecast_steps = 12
forecast = best_model.get_forecast(steps=forecast_steps)
pred_mean = forecast.predicted_mean
pred_ci = forecast.conf_int()

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(y, label='观测值')
plt.plot(pred_mean.index, pred_mean, label='预测值', color='red')
plt.fill_between(pred_ci.index, pred_ci.iloc[:,0], pred_ci.iloc[:,1], 
                 color='pink', alpha=0.3, label='95%置信区间')
plt.legend()
plt.title(f'ARIMA{best_order} 预测')
plt.show()