#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/23 16:20
@Author  : LI Zhe
"""
import numpy as np


def sgd(w, dw, config=None):
    """
    config: to store some hype_parameter
    - learning_rate : Scalar learning rate
    """
    if config is None:
        config = {}

    config['learning_rate'] = config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    config: to store some hype_parameter
    - learning_rate : Scalar learning rate
    - momentum: easy to understand
    - velocity: to store a moving average of the gradient
    """
    if config is None:
        config = {}

    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    v = config['momentum'] * v - config['learning_rate'] * dw
    config['velocity'] = v

    # next_w = None
    next_w = w + v

    return next_w, config


def RMSprop(x, dx, config=None):
    """
    config:
    - decay_rate:
    - learning_rate :
    - cache:
    - eps:
    """
    if config is None:
        config = {}

    config.setdefault('decay_rate', 0.99)
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('eps', 1e-8)
    config.setdefault('cache', np.zeros_like(x))

    config['cache'] = config['decay_rate'] * \
        config['cache'] + (1 - config['decay_rate']) * (dx ** 2)
    next_x = x - config['learning_rate'] * dx / \
        (np.sqrt(config['cache']) + config['eps'])

    return next_x, config


def Adam(x, dx, config=None):
    """
    config:
    - learning_rate :
    - beta1:
    - beta2:
    - eps:
    - m:
    - v:
    - t:
    """
    if config is None:
        config = {}

    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('eps', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 0)

    config['t'] += 1
    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dx
    config['v'] = config['beta2'] * config['v'] + \
        (1 - config['beta2']) * (dx ** 2)
    mb = config['m'] / (1 - config['beta1'] ** config['t'])
    vb = config['v'] / (1 - config['beta2'] ** config['t'])
    next_x = x - config['learning_rate'] * mb / (np.sqrt(vb) + config['eps'])

    return next_x, config
