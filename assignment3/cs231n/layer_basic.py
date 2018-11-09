#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/23 16:23
@Author  : LI Zhe
"""
import numpy as np


def affine_forward(x, w, b):
    """
    computes a forward pass for a fully_connected network
    :param x: input data (N,d_1,d_2,d_3), what we used in the partial is (N, 32, 32, 3)
    :param w: weight (D, M) (M is the size of output)
    :param b: bias (M,)
    :return:
    - out : ouput (N, M)
    - cache : to store (x, w, b)
    """
    N = x.shape[0]
    x_reshape = x.reshape(N, -1)
    out = np.dot(x_reshape, w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    compute a backward pass for a fully_connected network
    :param dout: (N, M)
    :param cache: (x , w, b)
    :return:
        - dx: x like shape(N, d_1, d_2, d_3)
        - dw: w like shape(D, M)
        - db: b like shape(M,)
    """
    x, w, b = cache
    x_reshape = x.reshape(x.shape[0], -1)  # (N, D)

    dx = dout.dot(w.T)  # (N,M) * (M, D) = (N, D)
    dx = dx.reshape(*x.shape)  # (N, d_1, d_2, d_3)

    dw = x_reshape.T.dot(dout)  # (D, M) = (D, N) * (N, M)

    db = np.sum(dout.T, axis=1)  # (M,) = sum(M,N)
    return dx, dw, db


def relu_forward(x):
    out = x * (x >= 0)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    x = cache
    """
    if x >= 0:
        dx = dout
    else:
        dx = 0
    """
    dx = dout * (x >= 0)
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    a forward pass for batch normalization
    :param x: shape of (N, D)
    :param gamma: shape of (D,)
    :param beta: shape of (D,)
    :param bn_param:
            - mode : train or test
            - eps : 1e-8
            - momrntum : Constant for running mean / variance
            - running_mean
            - running_var
    :return:
        - out :shape of (N, D)
        - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get(
        'running_mean', np.zeros(
            D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
    out, cache = None, None

    if mode == 'train':    
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        xhat = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * xhat + beta
        cache = (mode, gamma, x, sample_mean, sample_var, eps, xhat)

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
#         print(sample_mean[0],running_mean[0])
        running_var = momentum * running_var + (1 - momentum) * sample_var
#         print(sample_var[0],running_var[0])
    elif mode == 'test':
#         scale = gamma / np.sqrt(running_var + eps)
#         out = x * scale + (beta - running_mean * scale)
#         print(scale[0])
        test_var = np.sqrt(running_var + eps)
        xhat = (x - running_mean) / test_var
        out = gamma * xhat + beta
        cache = (mode, gamma, x, test_var, xhat)
#         print(gamma[0], beta[0], running_mean[0], running_var[0], out[0])
    else:
        raise ValueError('Invalid forward batch_norm mode "%s"' % mode)

    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    create a backward pass for batch normalization
    :param dout: shape same as x（N, D）
    :param cache: (gamma, x, sample_mean, sample_var, eps, xhat)
    :return:
        - dx : shape of (N, D)
        - dgamma: shape of (D,)
        - dbeta: shape of (D,)
    """
    mode = cache[0]
    if mode == 'train':
        mode, gamma, x, sample_mean, sample_var, eps, xhat = cache
        N, D = x.shape

        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum(dout * xhat, axis=0)

        dxhat = dout * gamma

        dx3 = ((sample_var + eps) ** -0.5) * dxhat
        dvar_tmp = np.sum((x - sample_mean) * dxhat, axis=0)
        dvar = (-0.5) * ((sample_var + eps) ** -1.5) * dvar_tmp
        un_sum_dmean2 = (-1) * dx3

        dx2_tmp = 2 * dvar * (x - sample_mean)
        dx2 = np.ones_like(x) / N * dx2_tmp
        un_sum_dmean1 = (-1) * dx2

        un_sum_dmean = un_sum_dmean1 + un_sum_dmean2
        dmean = np.sum(un_sum_dmean, axis=0)
        dx1 = np.ones_like(x) / N * dmean
        dx = dx1 + dx2 + dx3
    elif mode == 'test':
        mode, gamma, x, test_var, xhat = cache
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(xhat * dout, axis=0)
        dxhat = gamma * dout
        dx = dxhat / test_var
    else:
        raise ValueError('Invalid forward batch_norm mode "%s"' % mode)    
        
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    gamma, x, sample_mean, sample_var, eps, xhat = cache
    N, D = x.shape

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * xhat, axis=0)

    dxhat = dout * gamma
    dvar = np.sum((x - sample_mean) * dxhat * (-0.5)
                  * np.power(sample_var + eps, -1.5), axis=0)
    dmean = np.sum((-1) * dxhat / np.sqrt(sample_var + eps),
                   axis=0) + dvar * np.mean(-2 * (x - sample_mean), axis=0)
    dx = 1 / np.sqrt(sample_var + eps) * dxhat + 2 * \
        dvar * (x - sample_mean) / N + dmean / N

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    performs the forward pass for (inverted) dropout
     :param x: input data
    :param dropout_param:
        - mode:'test' or 'train'
        - p : probability
        - seed: seed fot the random number generator
    :return:
        - out: Array of the shape of x
        - cache:
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])
    out, mask = None, None
    if mode == 'train':
        mask = (np.random.rand(*x.shape) >= p) / (1 - p)
        out = x * mask
    elif mode == 'test':
        out = x
    else:
        raise ValueError('Invalid forward dropout mode "%s"' % mode)

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)
    return out, cache


def dropout_backward(dout, cache):
    """
    a backward pass for dropout
    :param dout: shape of x
    :param cache:
    :return:
        dx: shape of x
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    if mode == 'train':
        dx = dout * mask
    elif mode == 'test':
        dx = dout
    else:
        raise ValueError('Invalid forward dropout mode "%s"' % mode)

    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    the forward pass for convolutional layer
    :param x: (N, C, H, W)
    :param w: (F, C, HH. WW)
    :param b: (F, )
    :param conv_param:
       - stride: S
       - pad: P
    :return:
       - out: (N, F, H_new, W_new)
       - cache: used in conv_backward (x, w, b, conv_param)
    """
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    stride, pad = conv_param['stride'], conv_param['pad']

    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, H_out, W_out))

    # padding for the dimension(H, W) in x(N, C, H, W)
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)),
                   mode='constant', constant_values=0)

    for i in range(H_out):
        for j in range(W_out):
            x_pad_mask = x_pad[:, :, i * stride:i *
                               stride + HH, j * stride: j * stride + WW]
            for k in range(F):
                out[:, k, i, j] = np.sum(
                    x_pad_mask * w[k, :, :, :], axis=(1, 2, 3))

    out = out + (b)[None, :, None, None]
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    a backward pass for convolutional layer
    :param dout: (N, F, H_out, W_out)
    :param cache: (x, w, b, con_param)
    :return:
        dx: shape of (N, C, W, H)
        dw: (F, C, WW, HH)
        db: (F,)
    """
    x, w, b, conv_param = cache
    stride, pad = conv_param['stride'], conv_param['pad']

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)),
                   mode='constant', constant_values=0)
    # dx = np.zeros_like(x)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)

    db = np.sum(dout, axis=(0, 2, 3))

    for i in range(H_out):
        for j in range(W_out):
            x_pad_mask = x_pad[:, :, i * stride:i *
                               stride + HH, j * stride: j * stride + WW]
            for k in range(F):
                dw[k, :, :, :] += np.sum((dout[:, k, i, j])[:, None, None, None] * x_pad_mask, axis=0)
            for n in range(N):
                dx_pad[n,
                        :,
                        i * stride: i * stride + HH,
                        j * stride: j * stride + WW] += np.sum((dout[n,
                            :,
                            i,
                            j])[:,
                                None,
                                None,
                                None] * w,
                            axis=0)

    dx = dx_pad[:, :, pad: -pad, pad: -pad]
    return dx, dw, db


def max_pool_forward_naive(x, max_pool_param):
    """
    a forward pass for max_pool
    :param x: (N, C< H, W)
    :param max_pool_param:
        - pool_height
        - pool_width
        - pool_stride
    :return:
        - out: (N, C, H_out, W_out)
        - cache:
            - x:
            - max_pool_param:
    """
    N, C, H, W = x.shape
    HH, WW, stride = max_pool_param['pool_height'], max_pool_param['pool_width'], max_pool_param[
        'pool_stride']

    H_out = 1 + (H - HH) // stride
    W_out = 1 + (W - WW) // stride
    out = np.zeros((N, C, H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            x_mask = x[:, :, i * stride:i * stride +
                       HH, j * stride: j * stride + WW]
            out[:, :, i, j] = np.max(x_mask, axis=(2, 3))

    cache = (x, max_pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    a backward pass for max pool layer
    :param dout: (N, C, H_out, W_out)
    :param cache: (x, max_pool_param)
    :return:
        dx: shape of (N, C, W, H)
    """
    x, max_pool_param = cache

    N, C, H, W = x.shape
    HH, WW, stride = max_pool_param['pool_height'], max_pool_param['pool_width'], max_pool_param[
        'pool_stride']

    H_out = 1 + (H - HH) // stride
    W_out = 1 + (W - WW) // stride

    dx = np.zeros_like(x)

    for i in range(H_out):
        for j in range(W_out):
            x_mask = x[:, :, i * stride:i * stride +
                       HH, j * stride: j * stride + WW]
            max_mask = np.max(x_mask, axis=(2, 3))
            temp_binary_mask = (x_mask == (max_mask[:, :, None, None]))
            dx[:, :, i * stride: i * stride + HH, j * stride:j * stride +
                WW] += (dout[:, :, i, j])[:, :, None, None] * temp_binary_mask
    return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    batch normalization for convolutional network
    :param x: (N,C,H,W)
    :param gamma: (C,)
    :param beta: (C,)
    :param bn_param:
        - mode : 'train' or 'test'
        - momentum :
        - running_mean:
        - running_var:
    :return:
        dout:
        cache:
    """
    N, C, H, W = x.shape

    temp_out, cache = batchnorm_forward(x.transpose(0, 3, 2, 1).reshape((N * H * W, C)), gamma, beta, bn_param)
    out = temp_out.reshape(N, W, H,  C).transpose(0, 3, 2, 1)

    return out, cache

def spatial_batchnorm_backward(dout, cache):

    N, C, H, W = dout.shape
    temp_dx, dgamma, dbeta = batchnorm_backward(dout.transpose(0, 3, 2, 1).reshape((N * H * W, C)), cache)
    dx = temp_dx.reshape(N, W, H,  C).transpose(0, 3, 2, 1)

    return dx, dgamma, dbeta

def svm_loss(x, y):
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    shift_probs = x - np.max(x, axis=1).reshape(-1,1)
    probs = np.exp(shift_probs) / np.sum(np.exp(shift_probs), axis=1).reshape(-1,1)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y] + 1e-10)) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
