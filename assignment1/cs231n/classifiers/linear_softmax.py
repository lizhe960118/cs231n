#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/15 18:18
@Author  : LI Zhe
"""
import numpy as np
def softmax_loss_naive(W, X, y, reg):
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0
    for i in range(num_train):
        scores = X[i].dot(W)  # 1 * C
        shift_scores = scores - max(scores)
        loss_i = - shift_scores[y[i]] + np.log(np.sum(np.exp(shift_scores)))
        loss += loss_i
        for j in range(num_classes):
            softmax_out = np.exp(shift_scores[j]) / sum(np.exp(shift_scores))
            if j == y[i]:
                dW[:, j] += (-1 + softmax_out) * X[i]
            else:
                dW[:, j] += softmax_out * X[i]
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW = dW / num_train + reg * W
    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    # dW = np.zeros(W.shape)  # D * C

    num_classes = W.shape[1]
    num_train = X.shape[0]
    # loss = 0.0
    scores = X.dot(W)  # N * C
    shift_scores = scores - np.max(scores, axis=1).reshape(-1, 1)
    softmax_output = np.exp(shift_scores) / np.sum(np.exp(shift_scores),axis=1).reshape(-1, 1)
    row_loss_sum = - np.sum(np.log(softmax_output[range(num_train), list(y)]))
    # row_loss_sum = - shift_scores[range(num_train), list(y)] + np.log(sum(np.exp(shift_scores)))
    loss = row_loss_sum / num_train + 0.5 * reg * np.sum(W * W)

    dS = softmax_output.copy()
    dS[range(num_train), list(y)] += -1
    dW = X.T.dot(dS)
    dW = dW / num_train + reg * W
    return loss, dW
