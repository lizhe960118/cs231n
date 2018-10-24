#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/15 8:58
@Author  : LI Zhe
"""
import os
import pickle
import numpy as np

def load_CIFAR10_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
        Y = np.array(Y)
        return X, Y
def load_CIFAR10(root):
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(root, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR10_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtrain = np.concatenate(xs)
    Ytrain = np.concatenate(ys)
    del X, Y
    Xtest, Ytest = load_CIFAR10_batch(os.path.join(root, 'test_batch'))
    return Xtrain, Ytrain, Xtest, Ytest
