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
    if smooth:
        x_new = np.linspace(min(bpp), max(bpp), 300)
        spl = make_interp_spline(bpp, psnr, k=2)  # k=2 for quadratic spline
        y_new = spl(x_new)
        plt.plot(x_new, y_new, color=color, linestyle=linestyle, marker=marker, linewidth=linewidth+1, label=label)
    else:
        plt.plot(bpp, psnr, color=color, linestyle=linestyle, marker=marker, linewidth=linewidth, label=label)

setup_plot()

# 数据及曲线配置
data = {
    # 'DCVC-RT(GOP-1)': {
    #     'bpp': [0.001372, 0.002439, 0.004409, 0.00804, 0.015002, 0.020473, 0.027858, 0.03776, 0.050744, 0.060526],
    #     'psnr': [33.7266, 35.508044, 37.112466, 38.544034, 39.917219, 40.577007, 41.218316, 41.838081, 42.424566, 42.758384],
    #     'color': 'hotpink', 'linestyle': '-', 'marker': 'o', 'smooth': False  # 使用深蓝色并去掉标记符号
    # },
    
    'DCVC-RT(GOP=1)': {
        'bpp': [0.021211, 0.030293, 0.033165, 0.03626, 0.03952, 0.041513, 0.043253, 0.047414, 0.049443, 0.067449],
        'psnr':  [33.9595, 35.329465, 35.686166, 36.021525, 36.363911, 36.543798, 36.719053, 37.059097, 37.236807, 38.437472],
        'color': 'hotpink', 'linestyle': '-', 'marker': 'o', 'smooth': False  # 使用深蓝色并去掉标记符号
    },
    # 'DCVC-DC(GOP32)': {
    #     'bpp': [0.012962, 0.019753, 0.031386, 0.048604],
    #     'psnr': [38.970087, 40.139259, 41.357272, 42.431407],
    #     'color': 'yellowgreen', 'linestyle': '-', 'marker': 'o', 'smooth': False  # 使用深蓝色并去掉标记符号
    # },

    # 'DCVC-FM(GOP-1)': {
    #     'bpp': [0.000999, 0.001592, 0.002786, 0.005256, 0.010522, 0.015025, 0.021277, 0.029731, 0.040974, 0.049337],
    #     'psnr': [32.233018, 34.116707, 35.918718, 37.627153, 39.292501, 40.101375, 40.879278, 41.611749, 42.295228, 42.673237],
    #     'color': 'blue', 'linestyle': '-', 'marker': 'o', 'smooth': False  # 使用深蓝色并去掉标记符号
    # },
    # 'RT-VQ': {
    #     'bpp': [0.000683, 0.001937, 0.005491, 0.012735],
    #     'psnr':  [27.994089, 32.330469, 35.64861, 37.159294],
    #     'color': '#3DAD33', 'linestyle': '-', 'marker': '', 'smooth': False  # 使用深蓝色并去掉标记符号
    # },

    'Mine-RD(GOP=1)': {
        'bpp': [0.016316, 0.023952, 0.026161, 0.028759, 0.031494, 0.032876, 0.034438, 0.037634, 0.039341, 0.054494],
        'psnr': [31.741689, 33.284336, 33.671774, 34.017783, 34.385556, 34.568333, 34.758819, 35.124814, 35.306883, 36.600532],
        'color': 'red', 'linestyle': '-', 'marker': 'o', 'smooth': False  # 使用红色
    },
    # 'Mine-GAN(GOP=1)': {
    #     'bpp':  [0.043211, 0.042292, 0.042925, 0.044305, 0.045015, 0.045353, 0.046294, 0.052481, 0.060647, 0.081531],
    #     'psnr': [33.01779, 33.003038, 33.063794, 33.210923, 33.325168, 33.39659, 33.523755, 33.204369, 33.065696, 32.944675],
    #     'color': 'black', 'linestyle': '-', 'marker': 'o', 'smooth': False  # 使用黑色
    # },      
}

# bd_rate = {}
# 绘制所有曲线
for label, params in data.items():
    plot_curve(params['bpp'], params['psnr'], label, params['color'], params['linestyle'], params['marker'], smooth=params.get('smooth', False))
    # bd_rate[label] = calculate_bdrate([1920*1080*120*bpp/1000.0 for bpp in params['bpp']], params['psnr'], [1920*1080*120*bpp/1000.0 for bpp in data['HM-16.26']['bpp']], data['HM-16.26']['psnr'])
    print(label, [1920*1080*120*bpp/1000.0 for bpp in params['bpp']])
    print(label, params['psnr'])
    
plt.legend(loc=4)
plt.savefig('/workspace/yx/DCVC/RT/uvg_psnr_yuv.pdf', bbox_inches='tight')

# for k, v in bd_rate.items():
#     print(k, v)