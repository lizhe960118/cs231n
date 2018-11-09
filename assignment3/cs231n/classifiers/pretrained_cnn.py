#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/9 10:43
@Author  : LI Zhe
"""
import h5py
import numpy as np

# from layer_basic import *
# from layer_faster import *
# from layer_utils import *

from cs231n.layer_basic import *
from cs231n.layer_faster import *
from cs231n.layer_utils import *

class PretrainedCNN(object):
    def __init__(self, dtype=np.float32, num_classes=100, input_size=64, h5_file=None):
        self.dtype = dtype
        self.conv_params = []
        self.input_size = input_size
        self.num_classes = num_classes

        self.conv_params.append({'stride': 2, 'pad': 2})
        self.conv_params.append({'stride': 1, 'pad': 1})
        self.conv_params.append({'stride': 2, 'pad': 1})
        self.conv_params.append({'stride': 1, 'pad': 1})
        self.conv_params.append({'stride': 2, 'pad': 1})
        self.conv_params.append({'stride': 1, 'pad': 1})
        self.conv_params.append({'stride': 2, 'pad': 1})
        self.conv_params.append({'stride': 1, 'pad': 1})
        self.conv_params.append({'stride': 2, 'pad': 1})

        self.filter_sizes = [5, 3, 3, 3, 3, 3, 3, 3, 3]
        self.num_filters = [64, 64, 128, 128, 256, 256, 512, 512, 1024]
        hidden_dim = 512

        self.bn_params = []

        cur_size = input_size
        prev_dim = 3
        self.params = {}

        # convolution layers
        for i, (f, next_dim) in enumerate(zip(self.filter_sizes, self.num_filters)):
            fan_in  = f * f * prev_dim
            self.params['W%d' % (i + 1)] = np.sqrt(2.0 / fan_in) * np.random.randn(next_dim, prev_dim, f, f)
            self.params['b%d' % (i + 1)] = np.zeros(next_dim)
            self.params['gamma%d'% (i + 1)] = np.ones(next_dim)
            self.params['beta%d'% (i + 1)] = np.zeros(next_dim)
            self.bn_params.append({'mode': 'train'})
            prev_dim = next_dim
            if self.conv_params[i]['stride'] == 2:
                cur_size = cur_size // 2

        # fc layer
        fan_in = cur_size * cur_size * self.num_filters[-1]
#         print(len(self.num_filters), i)
        self.params['W%d' % (len(self.num_filters) + 1)] = np.sqrt(2.0 / fan_in) * np.random.randn(fan_in, hidden_dim)
        self.params['b%d' % (len(self.num_filters) + 1)] = np.zeros(hidden_dim)
        self.params['gamma%d' % (len(self.num_filters) + 1)] = np.ones(hidden_dim)
        self.params['beta%d' % (len(self.num_filters) + 1)] = np.zeros(hidden_dim)
        self.bn_params.append({'mode': 'train'})

        self.params['W%d' % (len(self.num_filters) + 2)] = np.sqrt(2.0 / hidden_dim) * np.random.randn(hidden_dim, num_classes)
        self.params['b%d' % (len(self.num_filters) + 2)] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

        if h5_file:
            self.load_weights(h5_file)

    def load_weights(self, h5_file, verbose=False):
        # TEST 
#         x = np.random.randn(1, 3, self.input_size, self.input_size)
#         y = np.random.randint(self.num_classes, size=1)
#         loss, grads = self.loss(x, y)

        with h5py.File(h5_file, 'r') as f:
            for k, v in f.items():
                v = np.asarray(v)
                if k in self.params:
                    if verbose:
                        print(k, v.shape, self.params[k].shape)
                    if v.shape == self.params[k].shape:
                        self.params[k] = v.copy()
                    elif v.T.shape == self.params[k].shape:
                        self.params[k] = v.T.copy()
                    else:
                        raise ValueError("shapes for %s do not match" % k)
                if k.startswith('running_mean'):
                    i = int(k[12:]) - 1
                    self.bn_params[i]['running_mean'] = v.copy()
                    if verbose:
                        print(k, v.shape)
                if k.startswith('running_var'):
                    i = int(k[11:]) - 1
                    self.bn_params[i]['running_var'] = v.copy()
                    if verbose:
                        print(k, v.shape)
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    def forward(self, X, start=None, end=None, mode='test'):
        X = X.astype(self.dtype)
        if start is None:
            start = 0
        if end is None:
            end = len(self.conv_params) + 1
        layer_caches = []

        prev_a = X
        for i in range(start, end+1):
            if 0 <= i < len(self.conv_params):
                w, b = self.params['W%d' % (i + 1)], self.params['b%d' % (i + 1)]
                gamma, beta = self.params['gamma%d'% (i + 1)], self.params['beta%d'% (i + 1)]
                conv_param = self.conv_params[i]
                bn_param = self.bn_params[i]
                bn_param['mode'] = mode
                next_a, cache = conv_bn_relu_forward(prev_a, w, b, gamma, beta, conv_param, bn_param)

            elif i == len(self.conv_params):
                w, b = self.params['W%d' % (i + 1)], self.params['b%d' % (i + 1)]
                gamma, beta = self.params['gamma%d'% (i + 1)], self.params['beta%d'% (i + 1)]
                bn_param = self.bn_params[i]
                bn_param['mode'] = mode
                next_a, cache = affine_bn_relu_forward(prev_a, w, b, gamma, beta, bn_param)

            elif i == len(self.conv_params) + 1:
                w, b = self.params['W%d' % (i + 1)], self.params['b%d' % (i + 1)]
                next_a, cache = affine_forward(prev_a, w, b)

            else:
                raise ValueError('Invalid layer index %d' % i)

            layer_caches.append(cache)
            prev_a = next_a

        out = prev_a
        cache = (start, end, layer_caches)
        return out, cache

    def backward(self, dout, cache):
        start, end, layer_caches = cache
        dnext_a = dout
        grads = {}

        for i in reversed(range(start, end+1)):
#             print(i, len(layer_caches))
            if i == len(self.conv_params) + 1:
                dprev_a, dw, db = affine_backward(dnext_a, layer_caches.pop())
                grads['W%d' % (i + 1)] = dw
                grads['b%d' % (i + 1)] = db

            elif i == len(self.conv_params):
                dprev_a, dw, db, dgamma, dbeta = affine_bn_relu_backward(dnext_a, layer_caches.pop())
                grads['W%d' % (i + 1)] , grads['b%d' % (i + 1)] = dw, db
                grads['gamma%d'% (i + 1)], grads['beta%d'% (i + 1)] = dgamma, dbeta

            elif 0 <= i < len(self.conv_params):
                dprev_a, dw, db, dgamma, dbeta = conv_bn_relu_backward(dnext_a, layer_caches.pop())
                grads['W%d' % (i + 1)] , grads['b%d' % (i + 1)] = dw, db
                grads['gamma%d'% (i + 1)], grads['beta%d'% (i + 1)] = dgamma, dbeta

            else:
                raise ValueError('Invalid layer index %d' % i)

            dnext_a = dprev_a

        dX = dnext_a
        return dX, grads

    def loss(self, X, y=None):
        mode = 'test' if y is None else 'train'
        scores, cache = self.forward(X, mode=mode)
        if mode == 'test':
            return scores
        print("start")
        loss, dscores = softmax_loss(scores, y)
        dX, grads = self.backward(dscores, cache)
        return loss, grads






