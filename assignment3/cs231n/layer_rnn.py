#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/5 22:00
@Author  : LI Zhe
"""
import numpy as np

def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """

    :param x: shape (N, D)
    :param prev_h: (N, H)
    :param Wx: (D, H)
    :param Wh: (H, H)
    :param b: (H, 1)
    :return:
        - next_h：（N, H）
    """
    next_h = np.tanh(x.dot(Wx) + prev_h.dot(Wh) + b)
    cache = (x, prev_h, Wx, Wh, b, next_h)
    return next_h, cache

def rnn_step_backward(dnext_h, cache):
    """

    :param dnext_h: (N, H)
    :param cache:
    :return:
    """
    x, prev_h, Wx, Wh, b, next_h = cache
    dtheta = dnext_h * (1 - next_h ** 2) # (N ,H)
    db = np.sum(dtheta, axis=0) # (H, ) shape like dtheta[0]
    dWx = x.T.dot(dtheta) # (D, H) = (D, N) *(N, H)
    dWh = prev_h.T.dot(dtheta)# (H, H) = (H, N) * (N ,H)
    dprev_h = dtheta .dot(Wh.T)# (N, H) = (N, H) * (H, H)
    dx = dtheta.dot(Wx.T) # (N ,D) = (N, H) * (H, D)
    return dx, dprev_h, dWx, dWh, db

def rnn_forward(x, h0, Wx, Wh, b):
    N, T, D = x.shape
    (H, ) = b.shape
    prev_h = h0
    h = np.zeros((N, T, H))
    for t in range(T):
        xt = x[:, t, :]
        next_h, _ = rnn_step_forward(xt, prev_h, Wx, Wh, b)
        prev_h = next_h
        h[:, t, :] = next_h
    cache = (x, h0, Wx, Wh, b, h)
    return h, cache

def rnn_backward(dh ,cache):
    x, h0, Wx, Wh, b, h = cache
    N, T, H = h.shape
    _, _, D = x.shape

    next_h = h[:, T - 1,:]
    dprev_h = np.zeros((N, H))
    dx = np.zeros(x.shape)
    dWx = np.zeros(Wx.shape)
    dWh = np.zeros(Wh.shape)
    db = np.zeros(b.shape)

    for t in range(T - 1, -1, -1):
        xt = x[:, t, :]

        if t == 0:
            prev_h = h0
        else:
            prev_h = h[:, t-1, :]

        step_cache = (xt, prev_h, Wx, Wh, b, next_h)
        next_h = prev_h

        dnext_h = dh[:, t, :] + dprev_h
        dx[:, t, :], dprev_h, dWxt, dWht, dbt = rnn_step_backward(dnext_h, step_cache)
        dWx, dWh, db = dWx + dWxt, dWh + dWht, db + dbt

    dh0 = dprev_h
    return dx, dh0, dWx, dWh, db

def word_embedding_forward(x, W):
    """

    :param x: (N, T) N is mini_batch size, T is sequence length
    :param W: (V, D) V words, D dimension (W[0] is D dimension)
    :return:
        - out : (N, T, D)giving word vectors for all input words
        - cache
    """
    N, T = x.shape
    V, D = W.shape
    out = np.zeros((N, T, D))

    for i in range(N):
        for j in range(T):
            out[i, j] = W[x[i, j]]

    cache = (x, W.shape)
    return out, cache

def word_embedding_backward(dout, cache):
    x, W_shape = cache
    dW = np.zeros(W_shape) # (V, D)
    np.add.at(dW, x, dout) # x(N, T) dout(N, T, D)
    return dW

def sigmoid(x):
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

def temporal_affine_forward(x, w, b):
    """

    :param x:
    :param w:
    :param b:
    :return:
        -dout :(N, T, M)
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = (x, w, b, out)
    return out, cache

def temporal_affine_backward(dout, cache):
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))
    return dx, dw, db

def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    softmax for rnn
    :param x: (N, T, V)
    :param y: (N, T) 0 <= y[i. t] < V
    :param mask: (N, T)
    :param verbose:
    :return:
        - loss:
        - dx
    """
    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = - np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N

    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose:
        print('dx_flat', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)
    return loss, dx

def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """

    :param x: (N, D)
    :param prev_h: (N, H)
    :param prev_c: (N, H)
    :param Wx: (D, 4H)
    :param Wh:(H, 4H)
    :param b:(4H, )
    :return:
        - next_h: (N, H)
        - next_c: (N, H)
        - cache
    """
    H = Wh.shape[0]

    z = x.dot(Wx) + prev_h.dot(Wh) + b

    z_i = sigmoid(z[:, :H])
    z_f = sigmoid(z[:, H:2 * H])
    z_o = sigmoid(z[:, 2 * H:3 * H])
    z_g = np.tanh(z[:, 3 * H:])

    next_c = z_f * prev_c + z_i * z_g
    c_t = np.tanh(next_c)
    next_h = z_o * c_t

    cache = (z_i, z_f, z_o, z_g, c_t, prev_c, prev_h, Wx, Wh, x)

    return next_h, next_c, cache

def lstm_step_backward(dnext_h, dnext_c, cache):

    z_i, z_f, z_o, z_g, c_t, prev_c, prev_h, Wx, Wh, x = cache
    H = dnext_h.shape[1]

    dc_t = dnext_h * z_o * (1 - c_t * c_t) + dnext_c
    dz_o = c_t * dnext_h
    dz_f = prev_c * dc_t
    dz_i = z_g * dc_t
    dz_g = z_i * dc_t
    dprev_c = z_f * dc_t

    dz_o = dz_o * z_o * (1 - z_o)
    dz_f = dz_f * z_f * (1 - z_f)
    dz_i = dz_i * z_i * (1- z_i)
    dz_g = dz_g * (1 - z_g * z_g )

    dz = np.hstack((dz_i, dz_f, dz_o, dz_g))

    dWx = x.T.dot(dz)
    dWh = prev_h.T.dot(dz)

    db = np.sum(dz, axis=0)
    dx = dz.dot(Wx.T)
    dprev_h = dz.dot(Wh.T)

    return dx, dprev_h, dprev_c, dWx, dWh, db

def lstm_forward(x, h0, Wx, Wh, b):
    N, T, D = x.shape
    H  = b.shape[0] // 4
    h = np.zeros((N, T, H))
    prev_h = h0
    prev_c = np.zeros((N, H))
    cache = {}

    for t in range(T):
        xt = x[:, t, :]
        next_h, next_c, cache[t] = lstm_step_forward(xt, prev_h, prev_c, Wx, Wh, b)
        prev_h = next_h
        prev_c = next_c
        h[:, t, :] = next_h

    return h, cache

def lstm_backward(dh, cache):
    N, T, H = dh.shape
    z_i, z_f, z_o, z_g, c_t, prev_c, prev_h, Wx, Wh, x = cache[T-1]
    D = x.shape[1]

    dprev_h = np.zeros(prev_h.shape)
    dprev_c = np.zeros(prev_c.shape)
    dx = np.zeros((N, T, D))
    dWx = np.zeros(Wx.shape)
    dWh = np.zeros(Wh.shape)
    db = np.zeros((4 * H, ))

    for t in range(T - 1, -1, -1):
        step_cache = cache[t]
        dnext_h = dh[:, t, :] + dprev_h
        dnext_c = dprev_c
        dx[:, t, :], dprev_h, dprev_c, dWxt, dWht, dbt = lstm_step_backward(dnext_h, dnext_c,step_cache)
        dWx, dWh, db = dWx + dWxt, dWh + dWht, db + dbt

    dh0 = dprev_h
    return dx, dh0, dWx, dWh, db