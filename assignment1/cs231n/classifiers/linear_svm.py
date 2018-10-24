#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/15 17:07
@Author  : LI Zhe
"""
import numpy as np

def svm_loss_naive(W, X, y, reg):
    """
    linear svm, this function use loops
    :param W:  weights D * C
    :param X: train data N * D
    :param y: train labels (N, )
    :param reg: regularization strength
    :return: loss: as single float
              dw: gradient with respect to weights
    """
    dW = np.zeros(W.shape) # D * C

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W) # 1 * C
        correct_class_score = scores[y[i]] # 选出正确的这类的值
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i].T
                dW[:, y[i]] += -X[i].T
    loss /= num_train
    dW /= num_train

    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    return loss, dW

def svm_loss_vectorized(W, X, y, reg):
    # dW = np.zeros(W.shape)  # D * C

    num_classes = W.shape[1]
    num_train = X.shape[0]
    # loss = 0.0
    scores = X.dot(W)  # N * C
    correct_class_score = scores[range(num_train), list(y)].reshape(-1, 1)

    margin = scores - correct_class_score + 1
    margin = (margin > 0) * margin
    margin[range(num_train), list(y)] = 0.0
    row_loss_sum = np.sum(margin)
    loss = row_loss_sum / num_train + 0.5 * reg * np.sum(W * W)

    coeff_mat = np.zeros((num_train, num_classes))
    coeff_mat[margin > 0] = 1
    coeff_mat[range(num_train), list(y)] = 0
    coeff_mat[range(num_train), list(y)] = - np.sum(coeff_mat, axis=1)

    dW = X.T.dot(coeff_mat)
    dW = dW / num_train + reg * W
    return loss, dW
