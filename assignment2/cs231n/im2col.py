#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/2 11:10
@Author  : LI Zhe
"""

import numpy as np

def get_im2col_indices(x_shape, filter_height, filter_width, pad=1, stride=1):
    """
    计算输入图片需要转换的形状（out_height * out_width, filter_height * filter_width）
    :param x_shape:
    :param filter_height: 卷积核的高
    :param filter_width: 卷积核的宽
    :param pad:
    :param stride:
    :return:
    """
    N, C, H, W = x_shape

    assert ((H + 2 * pad - filter_height) % stride == 0)
    assert ((W + 2 * pad - filter_width) % stride == 0)
    out_height = (H + 2 * pad - filter_height) % stride + 1
    out_width = (W + 2 * pad - filter_width) % stride + 1

    i0 = np.repeat(np.arange(filter_height), filter_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(filter_width), filter_width * C)
    j1 = stride * np.repeat(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.tile(np.arange(C), filter_height * filter_width).reshape(-1, 1) #  filter_height * filter_width
    return (k, i, j)

def im2col_indices(x, filter_height, filter_width, pad=1, stride=1):
    p = pad
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    k, i, j = get_im2col_indices(x.shape, filter_height, filter_width, p, stride)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(filter_height * filter_width * C, -1)
    return cols

def col2im_indices(cols, x_shape, filter_height=3, filter_width=3, pad=1, stride=1):
    N, C, H, W = x_shape
    H_padded = H + 2 * pad
    W_padded = W + 2 * pad
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, filter_height, filter_width, pad, stride)
    cols_reshaped = cols.reshape(C * filter_height * filter_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if pad == 0:
        return x_padded
    return x_padded[:, :, pad:-pad, pad:-pad]
