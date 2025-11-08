import numpy as np
import torch


def rgb2yuv(rgb, out_range=1):
    """input: [0,255]
    """
    if isinstance(rgb, np.ndarray):
        rgb = rgb.copy()
    else:
        rgb = rgb.clone()

    # convert to [0, 255]
    if rgb.max() <= 1: rgb = 255 * rgb

    if isinstance(rgb, np.ndarray):
        yuv = rgb.copy()
    else:
        yuv = rgb.clone()


    yuv[:, 0, :, :] = 0.212600 * rgb[:, 0, :, :] + 0.715200 * rgb[:, 1, :, :] + 0.072200 * rgb[:, 2, :, :]
    yuv[:, 1, :, :] = -0.114572 * rgb[:, 0, :, :] - 0.385428 * rgb[:, 1, :, :] + 0.5 * rgb[:, 2, :, :] + 128.0
    yuv[:, 2, :, :] = 0.5 * rgb[:, 0, :, :] - 0.454153 * rgb[:, 1, :, :] - 0.045847 * rgb[:, 2, :, :] + 128.0
    yuv = yuv / 255.

    yuv = yuv * out_range

    if out_range > 1: yuv = yuv.round()

    return yuv


def yuv2rgb(yuv, out_range=1):
    """input: [0,1];    output: [0,1]
    """
    if isinstance(yuv, np.ndarray):
        yuv = yuv.copy()
    else:
        yuv = yuv.clone()

    # if yuv.max()>1: yuv = yuv / 255. # convert to [0, 1]

    yuv = yuv * 255.

    yuv[:, 0, :, :] = yuv[:, 0, :, :]
    yuv[:, 1, :, :] = yuv[:, 1, :, :] - 128
    yuv[:, 2, :, :] = yuv[:, 2, :, :] - 128

    if isinstance(yuv, np.ndarray):
        rgb = yuv.copy()
    else:
        rgb = yuv.clone()

    rgb[:, 0, :, :] = yuv[:, 0, :, :] + 1.57480 * yuv[:, 2, :, :]
    rgb[:, 1, :, :] = yuv[:, 0, :, :] - 0.18733 * yuv[:, 1, :, :] - 0.46813 * yuv[:, 2, :, :]
    rgb[:, 2, :, :] = yuv[:, 0, :, :] + 1.85563 * yuv[:, 1, :, :]

    rgb = rgb / 255
    rgb = rgb * out_range

    return rgb
