#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/24 16:27
@Author  : LI Zhe
"""
from cs231n.layer_basic import *
from cs231n.layer_utils import *
# from layer_basic import *
# from layer_utils import *


class TwoLayerNet(object):
    def __init__(
            self,
            input_size=32 * 32 * 3,
            hidden_size=100,
            output_size=10,
            weight_scale=1e-3,
            reg=0):
        self.params = {}
        self.reg = reg
        self.params['W1'] = weight_scale * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_scale * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None):

        # forward pass
        W1, b1 = self.params["W1"], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        ar1_out, ar1_cache = affine_relu_forward(X, W1, b1)
        a2_out, a2_cache = affine_forward(ar1_out, W2, b2)
        scores = a2_out

        if y is None:
            return scores
        # backward pass
        grads = {}
        loss, dS = softmax_loss(scores, y)
        loss = loss + 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        dx2, dw2, db2 = affine_backward(dS, a2_cache)
        grads['W2'] = dw2 + self.reg * W2
        grads['b2'] = db2

        dx1, dw1, db1 = affine_relu_backward(dx2, ar1_cache)
        grads['W1'] = dw1 + self.reg * W1
        grads['b1'] = db1

        return loss, grads


class FullyConnectedNet(object):

    def __init__(
            self,
            hidden_dims,
            input_size=32 * 32 * 3,
            output_size=10,
            dropout=0,
            use_batchnorm=False,
            weight_scale=1e-2,
            reg=0,
            dtype=np.float32,
            seed=None):
        """
        init the fc net
        :param self:
        :param hidden_dims: the number of hidden layers, a list such like [100, 10, 20]
        :param input_size:
        :param output_size:
        :param dropout:
        :param use_batchnorm:
        :param weight_scale:
        :param reg:
        :param dtype:
        :param seed:
        :return:
        """
        self.params = {}
        self.reg = reg
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype

        layer_input_dim = input_size
        for i, hd in enumerate(hidden_dims):
            self.params['W%d' % (i + 1)] = weight_scale * \
                np.random.randn(layer_input_dim, hd)
            self.params['b%d' % (i + 1)] = weight_scale * np.zeros(hd)
            if self.use_batchnorm:
                self.params['gamma%d' % (i + 1)] = np.ones(hd)
                self.params['beta%d' % (i + 1)] = np.zeros(hd)
            layer_input_dim = hd  # keep the last hidden layer size
        self.params['W%d' % self.num_layers] = weight_scale * \
            np.random.randn(layer_input_dim, output_size)
        self.params['b%d' % self.num_layers] = weight_scale * \
            np.zeros(output_size)

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed:
                self.dropout_param['seed'] = seed

        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for _ in range(self.num_layers - 1)]

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        if self.dropout_param:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        layer_input = X
        ar_cache = {}
        dp_cache = {}

        # forward pass
        for lay in range(self.num_layers - 1):
            if self.use_batchnorm:
                layer_input, ar_cache[lay] = affine_bn_relu_forward(layer_input,
                                                                    self.params['W%d'%(lay+1)],
                                                                    self.params['b%d'%(lay+1)],
                                                                    self.params['gamma%d'%(lay+1)],
                                                                    self.params['beta%d'%(lay+1)],
                                                                    self.bn_params[lay]
                                                                    )
            else:
                layer_input, ar_cache[lay] = affine_relu_forward(layer_input,
                                                                    self.params['W%d'%(lay+1)],
                                                                    self.params['b%d'%(lay+1)]
                                                                    )
            if self.use_dropout:
                layer_input, dp_cache[lay] = dropout_forward(layer_input, self.dropout_param)

        ar_out, ar_cache[self.num_layers] = affine_forward(layer_input,
                                                           self.params['W%d'%(self.num_layers)],
                                                           self.params['b%d'%(self.num_layers)]
                                                          )
        scores = ar_out

        if mode == 'test':
            return scores
        # backward pass
        grads = {}
        loss, dS = softmax_loss(scores, y)
        loss += 0.5 * self.reg * np.sum(self.params['W%d' % self.num_layers] * self.params['W%d' % self.num_layers])
        dhout, dw, db = affine_backward(dS, ar_cache[self.num_layers])
        grads['W%d'%self.num_layers] = dw + self.reg * self.params['W%d'%self.num_layers]
        grads['b%d'%self.num_layers] = db

        for idx in range(self.num_layers-1):
            lay = self.num_layers - 1 - idx - 1
            # idx = 0, lay = self.num_layers - 2, lay + 1 = self.num_layers - 1
            # idx = self.num_layers - 2, lay = 0
            loss += 0.5 * self.reg * np.sum(self.params['W%d' % (lay + 1)] * self.params['W%d' % (lay + 1)])
            if self.use_dropout:
                dhout = dropout_backward(dhout, dp_cache[lay])
            if self.use_batchnorm:
                dhout, dw, db, dgamma, dbeta = affine_bn_relu_backward(dhout, ar_cache[lay])
            else:
                dhout, dw, db = affine_relu_backward(dhout, ar_cache[lay])
            grads['W%d' % (lay + 1)] = dw + self.reg * self.params['W%d' % (lay + 1)]
            grads['b%d' % (lay + 1)] = db
            if self.use_batchnorm:
                grads['gamma%d' % (lay + 1)] = dgamma
                grads['beta%d' % (lay + 1)] = dbeta
        return loss, grads
