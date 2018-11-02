#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/2 11:56
@Author  : LI Zhe
"""
import numpy as np
from cs231n.im2col import *
# from im2col import *
try:
    from cs231n.im2col_cython import col2im_cython, im2col_cython
    from cs231n.im2col_cython import col2im_6d_cython
#     from im2col_cython import col2im_cython, im2col_cython
#     from im2col_cython import col2im_6d_cython
except ImportError:
    print('run the following from the cs231n directory and try again:')
    print('python setup.py build_ext --inplace')
    print('You may also need to restart your iPython kernel')

def conv_forward_im2col(x, w, b, conv_param):
    """
    使用im2col计算卷积的前向过程
    :param x:
    :param w:
    :param b:
    :param conv_param:
    :return:
    """
    N, C, H, W = x.shape
    num_filters, _ , filter_height, filter_width = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    assert ((H + 2 * pad - filter_height) % stride == 0, 'input height failed')
    assert ((W + 2 * pad - filter_width) % stride == 0, 'input width failed')
    out_height = (H + 2 * pad - filter_height) // stride + 1
    out_width = (W + 2 * pad - filter_width) // stride + 1
    out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

    x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
    # x_cols = im2col_cython(x, w.shape[2], w.shape[3], pad, stride)
    res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

    out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0]) # (F, O_h,O_w, N)
    out = out.transpose(3, 0, 1, 2)
    cache = (x, w, b, conv_param, x_cols)
    return out, cache

def conv_backward_im2col(dout, cache):
    """
    使用im2col计算卷积的反向过程
    :param dout:
    :param cache:
    :return:
    """
    x, w, b, conv_param, x_cols = cache
    stride, pad = conv_param['stride'], conv_param['pad']

    db = np.sum(dout, axis=(0, 2, 3))

    # N, C, H, W = x.shape
    num_filters, _ , filter_height, filter_width = w.shape
    # dout = (N, F, O_h,O_w)
    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)
    dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)
    dx_cols = w.reshape(num_filters, -1).T.dot(dout_reshaped)

    dx = col2im_indices(dx_cols, x.shape, filter_height, filter_width, pad, stride)
    # dx = col2im_cython(dx_cols, N, C, H, W, filter_height, filter_width, pad, stride)

    return dx, dw, db

def conv_forward_fast(x, w, b, conv_param):
    """
    实现和conv_forward_im2col相同的功能
    :param x:
    :param w:
    :param b:
    :param conv_param:
    :return:
    """
    N, C, H, W = x.shape
    num_filters, _ , filter_height, filter_width = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    assert ((H + 2 * pad - filter_height) % stride == 0, 'input height failed')
    assert ((W + 2 * pad - filter_width) % stride == 0, 'input width failed')
    # H, W更新了，在计算下面的out的时候注意！
    H += 2 * pad 
    W += 2 * pad
    out_height = (H - filter_height) // stride + 1
    out_width = (W - filter_width) // stride + 1
    
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    output_shape = (C, filter_height, filter_width, N, out_height, out_height)
    stride_shape = (H * W, W, 1, C * H * W, stride * W, stride)
    stride_shape = x.itemsize * np.array(stride_shape)
    x_stride = np.lib.stride_tricks.as_strided(x_padded, shape=output_shape, strides=stride_shape)
    x_cols = np.ascontiguousarray(x_stride)
    x_cols.shape = (C * filter_height * filter_width, N * out_height * out_width)

    res = w.reshape(num_filters, -1).dot(x_cols) + b.reshape(-1, 1)
    res.shape = (num_filters, N, out_height, out_width)
    out = res.transpose(1, 0, 2, 3)

    out = np.ascontiguousarray(out)
    cache = (x, w, b, conv_param, x_cols)

    return out, cache

def conv_backward_fast(dout, cache):
    # dout (N, F, out_height, out_width)
    x, w, b, conv_param, x_cols = cache

    N, C, H, W = x.shape
    num_filters, _ , filter_height, filter_width = w.shape
    _, _ ,out_height, out_width = dout.shape
    
    stride, pad = conv_param['stride'], conv_param['pad']
          
    dx = np.zeros(x.shape, dtype=x.dtype)
    
    db = np.sum(dout, axis=(0, 2, 3))

    dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(num_filters, -1)
    dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

    dx_cols = w.reshape(num_filters, -1).T.dot(dout_reshaped)
    dx_cols.shape = (C, filter_height, filter_width, N, out_height, out_width)
#     dx = col2im_cython(dx_cols, N, C, H, W, filter_height, filter_width, pad, stride)
    dx = col2im_6d_cython(dx_cols, N, C, H, W, filter_height, filter_width, pad, stride)

    return dx, dw, db

def max_pool_forward_im2col(x, pool_param):

    N, C, H, W = x.shape
    stride, pool_height, pool_width = pool_param['pool_stride'], pool_param['pool_height'], pool_param['pool_width']

    assert ((H - pool_height) % stride == 0, 'input height failed')
    assert ((W - pool_width) % stride == 0, 'input width failed')

    out_height = (H - pool_height) // stride + 1
    out_width = (W  - pool_width) // stride + 1

    x_split = x.reshape(N * C, 1, H, W)

    x_cols = im2col_indices(x_split, pool_height, pool_width, pad=0, stride=stride)
    # x_cols = im2col_cython(x_split, pool_height, pool_width, pad=0, stride=stride)
    x_cols_argmax = np.argmax(x_cols, axis=0)
    x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]

    out = x_cols_max.reshape(out_height, out_width, N, C)
    out = out.transpose(2, 3, 0, 1)
    cache = (x, pool_param, x_cols, x_cols_argmax)
    return out, cache

def max_pool_backward_im2col(dout, cache):

    x, pool_param, x_cols, x_cols_argmax = cache

    N, C, H, W = x.shape
    stride, pool_height, pool_width = pool_param['pool_stride'], pool_param['pool_height'], pool_param['pool_width']
    x_split = x.reshape(N * C, 1, H, W)
    dout_reshaped = dout.transpose(2, 3, 0, 1).flatten()
    dx_cols = np.zeros_like(x_cols)
    dx_cols[x_cols_argmax, np.arange(dx_cols.shape[1])] = dout_reshaped

    dx = col2im_indices(dx_cols, x_split.shape, pool_height, pool_width, pad=0, stride=stride)
    dx = dx.reshape(x.shape)
    return dx

def max_pool_forward_reshape(x, pool_param):

    N, C, H, W = x.shape
    stride, pool_height, pool_width = pool_param['pool_stride'], pool_param['pool_height'], pool_param['pool_width']

    assert ((H - pool_height) % stride == 0, 'input height failed')
    assert ((W - pool_width) % stride == 0, 'input width failed')

    x_reshaped = x.reshape(N, C, H // pool_height, pool_height, W // pool_width, pool_width)
    out = x_reshaped.max(axis=3).max(axis=4)

    cache = (x, x_reshaped, out)
    return out, cache

def max_pool_backward_reshape(dout, cache):

    x, x_reshaped, out = cache

    dx_reshaped = np.zeros_like(x_reshaped)
    out_new_axis = out[:, :, :, np.newaxis, :, np.newaxis]
    mask = (x_reshaped == out_new_axis)
    dout_new_axis = dout[:, :, :, np.newaxis, :, np.newaxis]
    dout_broadcast, _ = np.broadcast_arrays(dout_new_axis, dx_reshaped)
    dx_reshaped[mask] = dout_broadcast[mask]
    dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)

    dx = dx_reshaped.reshape(x.shape)
    return dx

def max_pool_forward_fast(x, pool_param):

    N, C, H, W = x.shape
    stride, pool_height, pool_width = pool_param['pool_stride'], pool_param['pool_height'], pool_param['pool_width']

    is_same_size = pool_height == pool_width == stride
    pool_flags = (H % pool_height == 0) and  (W % pool_width == 0)

    if is_same_size and pool_flags:
        out, reshape_cache = max_pool_forward_reshape(x, pool_param)
        cache = ('reshape_cache', reshape_cache)
    else:
        out, im2col_cache = max_pool_backward_im2col(x, pool_param)
        cache = ('im2col_cache', im2col_cache)

    return out, cache

def max_pool_backward_fast(dout, cache):

    category_name, method_cache = cache

    if category_name == "reshape_cache":
        return max_pool_backward_reshape(dout, method_cache)
    elif category_name == "im2col_cache":
        return max_pool_backward_im2col(dout, method_cache)
    else:
        raise ValueError('Invalid method "%s"' % category_name)