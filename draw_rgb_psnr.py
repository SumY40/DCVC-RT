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
    # 'x265': {
    #     'bpp': [0.015771428571428568, 0.029028571428571428, 0.052000000000000005, 0.09394285714285715],
    #     'psnr': [31.833688699360337, 33.43321961620469, 34.86494669509595, 36.14780383795309],
    #     'color': 'y', 'linestyle': '--', 'marker': '^'
    # },
    # 'HM-16.26': {
    #     'bpp': [0.011915847984641388, 0.016436000422545564, 0.022610246500220456, 0.03231771407444151, 0.04611868363278953, 0.06851267407039976, 0.10586786609898588, 0.18927628164498822],
    #     'psnr': [33.124265159879414, 33.90985329591093, 34.67173445437636, 35.47460635148344, 36.23948356934956, 36.9922902647938, 37.735295746298064, 38.6438886750312],
    #     'color': 'r', 'linestyle': '--', 'marker': 'v'
    # },
    # 'VTM-18.0': {
    #     'bpp': [0.1650409111368313, 0.09170101571685774, 0.06272838702417695, 0.04050920184817754, 0.029524148249191657, 0.01990065012308936, 0.014519469246031745, 0.010520419973544973],
    #     'psnr': [38.8901701746952, 38.023744899602164, 37.39964779998575, 36.605136100025405, 35.95015586132095, 35.1050653273151, 34.382187936987194, 33.63258020863646],
    #     'color': '#708090', 'linestyle': '--', 'marker': '<'
    # },
    # 'DVC-Pro': {
    #     'bpp': [0.030857142857142857, 0.04754285714285714, 0.06377142857142858, 0.10171428571428572],
    #     'psnr': [32.91014925373134, 34.14272921108742, 35.21876332622601, 36.32400852878465],
    #     'color': 'indigo', 'linestyle': '-', 'marker': 's'
    # },
    # 'MLVC': {
    #     'bpp': [0.05973841856106545, 0.07624247042300515, 0.11247474167827187, 0.15742708837425032],
    #     'psnr': [35.19741068924839, 36.03549830980149, 37.08771165935253, 38.07997945968074],
    #     'color': 'green', 'linestyle': '-', 'marker': 'h'
    # },
    # 'RLVC': {
    #     'bpp': [0.04222457810113268, 0.06169566488043191, 0.08760060437759638, 0.13656441862969096],
    #     'psnr': [33.1712667437323, 34.83342818638209, 36.00509248468054, 37.19760710013524],
    #     'color': 'orange', 'linestyle': '-', 'marker': '*'
    # },
    # 'DCVC': {
    #     'bpp': [0.026285714285714284, 0.039999999999999994, 0.06240000000000001, 0.10148571428571429],
    #     'psnr': [33.441705756929636, 34.67539445628998, 35.868528784648184, 36.98081023454158],
    #     'color': 'hotpink', 'linestyle': '-', 'marker': 'D'
    # },
    # 'VCT': {
    #     'bpp': [0.027978119, 0.045626774, 0.072500112, 0.140096193],
    #     'psnr': [34.51184725, 35.59384371, 36.58848815, 37.65634805],
    #     'color': 'peru', 'linestyle': '-', 'marker': 'p'
    # },
    # 'DCVC-DC': {
    #     'bpp': [0.023971143, 0.036314857, 0.058432857, 0.096979857],
    #     'psnr': [35.8289743, 36.8120284, 37.75125186, 38.61019459],
    #     'color': 'black', 'linestyle': '-', 'marker': '', 'smooth': True
    # },
    # 'DCVC-DC': {
    #     'bpp': [0.020627, 0.031428, 0.042630, 0.068784],
    #     'psnr': [35.640835, 38.133659, 39.283767, 40.303677],
    #     'color': 'black', 'linestyle': '-', 'marker': '', 'smooth': True
    # },
    # 'DHVC(GOP32)': {
    #     'bpp':  [0.03386053138092869, 0.04385626820993743, 0.06626904993953094, 0.100020034626423411],
    #     'psnr': [36.60731559079724, 37.1499021411, 37.886401104546, 38.6184353223456],
    #     'color': '#645231', 'linestyle': '-', 'marker': 'o'
    # },
    # 'DHVC(GOP-1)': {
    #     'bpp': [0.03546053138092869, 0.04665626820993743, 0.06816904993953094, 0.1060034626423411],
    #     'psnr': [36.60731559079724, 37.1499021411, 37.886401104546, 38.6184353223456],
    #     'color': '#ABCDEF', 'linestyle': '-', 'marker': 'o'
    # },
    # 'DCVC-FM(GOP-1)': {
    #     'bpp': [0.001180, 0.003849, 0.007496, 0.014917, 0.029474, 0.069175],
    #     'psnr': [28.768114, 32.269988, 33.859022, 35.323397, 36.639032, 38.008984],
    #     'color': '#2E33EB', 'linestyle': '-', 'marker': '', 'smooth': True  # 使用深蓝色并去掉标记符号
    # },
    # 'All intra': {
    #     'bpp': [0.041751, 0.080145, 0.148453, 0.278508],
    #     'psnr': [31.964035, 34.360542, 36.509648, 38.334808],
    #     'color': '#1E10BF', 'linestyle': '-', 'marker': '', 'smooth': True  # 使用深蓝色并去掉标记符号
    # },
    # 'Flow-Free(GOP-1)': {
    #     'bpp': [0.001407, 0.004691, 0.008841, 0.016865, 0.031645, 0.058096],
    #     'psnr': [29.414147, 32.726219, 34.096128, 35.373153, 36.529322, 37.516583],
    #     'color': '#49BBEB', 'linestyle': '-', 'marker': '', 'smooth': True  # 使用深蓝色并去掉标记符号
    # },
    # 'DCVC-RT(-1)': {
    #     'bpp':  [0.001257, 0.004526, 0.008662, 0.01667, 0.031429, 0.069353],
    #     'psnr': [29.414147, 32.726219, 34.096128, 35.373153, 36.529322, 37.778779],
    #     'color': '#7775EB', 'linestyle': '-', 'marker': '', 'smooth': True  # 使用深蓝色并去掉标记符号
    # },
    'Mine-RD(GOP=1)': {
        'bpp': [0.016316, 0.023952, 0.02616, 0.028759, 0.031494, 0.032875, 0.034437, 0.037634, 0.039342, 0.054494],
        'psnr':  [31.741624, 33.284295, 33.671716, 34.017852, 34.38553, 34.568475, 34.758768, 35.124835, 35.306962, 36.600592],
        'color': 'red', 'linestyle': '-', 'marker': 'o', 'smooth': False  # 使用红色
    },
    'DCVC-RT(GOP=1)': {
        'bpp': [0.021211, 0.030293, 0.033165, 0.03626, 0.03952, 0.041513, 0.043253, 0.047414, 0.049443, 0.067449],
        'psnr':  [33.9595, 35.329465, 35.686166, 36.021525, 36.363911, 36.543798, 36.719053, 37.059097, 37.236807, 38.437472],
        'color': 'hotpink', 'linestyle': '-', 'marker': 'o', 'smooth': False  # 使用深蓝色并去掉标记符号
    },



}

# bd_rate = {}
# 绘制所有曲线
for label, params in data.items():
    plot_curve(params['bpp'], params['psnr'], label, params['color'], params['linestyle'], params['marker'], smooth=params.get('smooth', False))
    # bd_rate[label] = calculate_bdrate([1920*1080*120*bpp/1000.0 for bpp in params['bpp']], params['psnr'], [1920*1080*120*bpp/1000.0 for bpp in data['HM-16.26']['bpp']], data['HM-16.26']['psnr'])
    print(label, [1920*1080*120*bpp/1000.0 for bpp in params['bpp']])
    print(label, params['psnr'])
    
plt.legend(loc=4)
plt.savefig('/workspace/yx/DCVC/RT/uvg_psnr.pdf', bbox_inches='tight')

# for k, v in bd_rate.items():
#     print(k, v)