#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/2 15:13
@Author  : LI Zhe
"""
import numpy as np

from cs231n.layer_basic import *
from cs231n.layer_utils import *
from cs231n.layer_faster import *
# from layer_basic import *
# from layer_utils import *
# from layer_faster import *

class ThreeLayerConvNet(object):

    def __init__(
            self,
            input_dim = (3, 32, 32),
            num_filters = 32,
            filter_size = 7,
            hidden_dim = 100,
            num_classes=10,
            weight_scale=1e-3,
            reg=0,
            dtype=np.float32):

        self.params = {}
        self.reg = reg
        self.dtype = dtype

        C, H, W = input_dim
        self.params['W1'] = weight_scale * \
            np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = weight_scale * \
            np.random.randn((H // 2) * (W //2) * num_filters, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * \
            np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):

        W1, b1 = self.params["W1"], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        filter_size = W1.shape[2]
        conv_param = {'stride':1, 'pad':(filter_size - 1)// 2}
        pool_param = {'pool_height': 2, 'pool_width':2, 'pool_stride': 2}

        scores = None

        # forward pass
        conv_forward_out_1, cache_forward_1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        affine_forward_out_2, cache_forward_2 = affine_relu_forward(conv_forward_out_1, W2, b2)
        scores, cache_forward_3 = affine_forward(affine_forward_out_2, W3, b3)

        if y is None:
            return scores

        grads = {}
        loss, dS = softmax_loss(scores, y)

        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))

        dx3, dw3, db3 = affine_backward(dS, cache_forward_3)
        grads['W3'] = dw3 + self.reg * W3
        grads['b3'] = db3

        dx2, dw2, db2 = affine_relu_backward(dx3, cache_forward_2)
        grads['W2'] = dw2 + self.reg * W2
        grads['b2'] = db2

        dx1, dw1, db1 = conv_relu_pool_backward(dx2, cache_forward_1)
        grads['W1'] = dw1 + self.reg * W1
        grads['b1'] = db1

        return loss, grads