import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, interp1d
from scipy.integrate import quad
import os

def calculate_bdrate(rate1, psnr1, rate2, psnr2):
    """
    Calculate the BD-Rate between two rate-distortion curves.
    
    Parameters:
    rate1, psnr1: Bitrates and PSNRs for the first configuration (lists or numpy arrays)
    rate2, psnr2: Bitrates and PSNRs for the second configuration (lists or numpy arrays)
    
    Returns:
    The BD-Rate, representing the percentage bit rate difference.
    """
    
    # Logarithm of bitrates for interpolation
    log_rate1 = np.log(rate1)
    log_rate2 = np.log(rate2)
    
    # Interpolate PSNR as a function of log bitrate
    p1 = interp1d(psnr1, log_rate1, fill_value="extrapolate", kind='cubic')
    p2 = interp1d(psnr2, log_rate2, fill_value="extrapolate", kind='cubic')
    
    # Integration range
    min_psnr = max(min(psnr1), min(psnr2))
    max_psnr = min(max(psnr1), max(psnr2))
    
    # Define functions for integration
    def f1(x):
        return np.exp(p1(x))
    
    def f2(x):
        return np.exp(p2(x))
    
    # Integrate the functions
    integral1, _ = quad(f1, min_psnr, max_psnr)
    integral2, _ = quad(f2, min_psnr, max_psnr)
    
    # Calculate average exponential difference
    avg_exp_diff = (integral2 - integral1) / (max_psnr - min_psnr)
    
    # Convert to percentage
    bd_rate = (np.exp(avg_exp_diff) - 1) * 100
    
    return bd_rate


# 绘图样式配置
def setup_plot(font_size=14):
    plt.rc('font', family='serif', weight='normal', size=font_size)
    plt.grid(True)
    plt.xlabel('Bpp')
    plt.ylabel('PSNR')
    plt.title('UVG')

# 绘制曲线
def plot_curve(bpp, psnr, label, color, linestyle, marker, linewidth=2, smooth=False):
    # 检查数据点数量是否足够进行平滑处理（至少3个点用于k=2样条）
    if smooth and len(bpp) >= 3:
        x_new = np.linspace(min(bpp), max(bpp), 300)
        spl = make_interp_spline(bpp, psnr, k=2)  # 确保数据点足够
        y_new = spl(x_new)
        plt.plot(x_new, y_new, color=color, linestyle=linestyle, marker=marker, linewidth=linewidth+1, label=label)
    else:
        plt.plot(bpp, psnr, color=color, linestyle=linestyle, marker=marker, linewidth=linewidth, label=label)
setup_plot()

# 数据及曲线配置
data = {
    # 'DCVC-RT(GOP-1)': {
    #     'bpp': [0.00137, 0.007999, 0.014906, 0.059815],
    #     'psnr': [30.663792, 35.553473, 36.943663, 39.768269],
    #     'color': '#49BBEB', 'linestyle': '-', 'marker': 'o', 'smooth': False  # 使用深蓝色并去掉标记符号
    # # },
    'RT_INTRA': {
        'bpp': [0.017696, 0.034394, 0.066966, 0.101781, 0.151695, 0.225225],
        'psnr': [30.208284, 32.802864, 35.40298, 37.066832, 38.632661, 40.078561],
        'color': '#54AADD', 'linestyle': '-', 'marker': 'o', 'smooth': False  # 使用深蓝色并去掉标记符号
    },
    'MY_INTRA': {
        'bpp':  [0.007714, 0.027301, 0.072117, 0.143694],
        'psnr':  [26.872398, 31.743222, 35.600088, 38.333791],
        'color': '#AAABEB', 'linestyle': '-', 'marker': 'o', 'smooth': False  # 使用深蓝色并去掉标记符号
    },
#     'New(1e-4)': {
#         'bpp':   [0.02747],
#         'psnr':   [36.631027],
#         'color': '#A421BF', 'linestyle': '-', 'marker': 'D', 'smooth': False  # 使用深蓝色并去掉标记符号
#     },
#     'RT_finetune(GOP-1)': {
#         'bpp':   [0.000742, 0.001643, 0.008055, 0.022475],
#         'psnr':  [25.11428, 29.713177, 34.783338, 37.252217],
#         'color': '#A1111F', 'linestyle': '-', 'marker': 'D', 'smooth': False  # 使用深蓝色并去掉标记符号
#     },
#     '7 1e-4':{
#         'bpp':   [0.000971, 0.002526, 0.00781, 0.02277],
#         'psnr':  [26.043998, 29.800821, 33.149527, 36.022704],
#         'color': 'lightcoral', 'linestyle': '-', 'marker': '', 'smooth': False  # 使用深蓝色并去掉标记符号
#     },
#     '7 5e-6':{
#         'bpp':   [0.000975, 0.001758, 0.006383, 0.021296],
#         'psnr':  [26.840376, 29.674456, 33.370917, 36.322206],
#         'color': '#458854', 'linestyle': '-', 'marker': 'D', 'smooth': False  # 使用深蓝色并去掉标记符号
#     },
#   'DCVC-RT(GOP32)': {
#         'bpp': [0.001835, 0.003128, 0.005412, 0.009517, 0.017105, 0.030693, 0.064733],
#         'psnr': [30.623003, 32.393617, 34.016366, 35.488452, 36.905425, 38.237877, 39.780539],
#         'color': '#555D48', 'linestyle': '-', 'marker': '', 'smooth': True  # 使用深蓝色并去掉标记符号
#     },


}

# bd_rate = {}
# 绘制所有曲线
for label, params in data.items():
    plot_curve(params['bpp'], params['psnr'], label, params['color'], params['linestyle'], params['marker'], smooth=params.get('smooth', False))
    # bd_rate[label] = calculate_bdrate([1920*1080*120*bpp/1000.0 for bpp in params['bpp']], params['psnr'], [1920*1080*120*bpp/1000.0 for bpp in data['HM-16.26']['bpp']], data['HM-16.26']['psnr'])
    print(label, [1920*1080*120*bpp/1000.0 for bpp in params['bpp']])
    print(label, params['psnr'])
    
plt.legend(loc=4)
plt.savefig('/workspace/lxc/DCVC/RT/uvg_psnr709.pdf', bbox_inches='tight')

# for k, v in bd_rate.items():
#     print(k, v)