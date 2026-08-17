import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ========== 中文字体设置 ==========
font_path = r'C:\Windows\Fonsts\msyh.ttc'   # 微软雅黑

try:
    chinese_font = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = chinese_font.get_name()
except:
    # 如果找不到文件，尝试用字体名称
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    
plt.rcParams['axes.unicode_minus'] = False
