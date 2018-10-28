#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/24 16:09
@Author  : LI Zhe
"""
from cs231n.layer_basic import *
# from layer_basic import *

def affine_relu_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    a, fc_cache = affine_forward(x, w, b)
    bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bn)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

def affine_bn_relu_backward(dout, cache):
    fc_cache, bn_cache, relu_cache = cache
    dbn = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = batchnorm_backward(dbn, bn_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta

def conv_relu_forward_naive(x, w, b, conv_param):
    c, conv_cache = conv_forward_naive(x, w, b, conv_param)
    out, relu_cache = relu_forward(c)
    cache = (conv_cache, relu_cache)
    return out, cache

def conv_relu_backward_naive(dout, cache):
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_naive(da, conv_cache)
    return dx, dw, db

def conv_relu_pool_forward_naive(x, w, b, conv_param, max_pool_param):
    c, conv_cache = conv_forward_naive(x, w, b, conv_param)
    cr, relu_cache = relu_forward(c)
    out, max_pool_cache = max_pool_forward_naive(cr, max_pool_param)
    cache = (conv_cache, relu_cache, max_pool_cache)
    return out, cache

def conv_relu_pool_backward_naive(dout, cache):
    conv_cache, relu_cache, max_pool_cache = cache
    dmp = max_pool_backward_naive(dout, max_pool_cache)
    da = relu_backward(dmp, relu_cache)
    dx, dw, db = conv_backward_naive(da, conv_cache)
    return dx, dw, db

