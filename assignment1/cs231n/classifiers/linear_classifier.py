#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/16 16:10
@Author  : LI Zhe
"""
from cs231n.classifiers.linear_softmax import *
from cs231n.classifiers.linear_svm import *
import numpy as np

class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def loss(self, X_batch, y_batch, reg=0.0):
        pass

    def predict(self, X):
        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis=1)
        return y_pred

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        num_train, dim = X.shape
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        loss_history = []

        for it in range(num_iters):
            idx = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[idx]
            y_batch = y[idx]

            loss, grads = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # Update the weights
            self.W += - learning_rate * grads

            if verbose and it % 100 == 0:
            #   print(type(loss))
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

class LinearSVM(LinearClassifier):
    def loss(self, X_batch, y_batch, reg=0.0):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)

class LinearSoftmax(LinearClassifier):
    def loss(self, X_batch, y_batch, reg=0.0):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
