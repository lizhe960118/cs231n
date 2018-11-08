#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/7 11:34
@Author  : LI Zhe
"""

import numpy as np
from cs231n import optimizer
# import optimizer
# from coco_utils import sample_coco_minibatch
from cs231n.coco_utils import sample_coco_minibatch

class CaptioningSolver(object):
    def __init__(self, model, data, **kwargs):
        self.model = model
        self.data = data

        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_param = kwargs.pop('optim_param', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)

        self.p_num = kwargs.pop('p_num', 10)
        self.verbose = kwargs.pop('verbose', True)

        # throw a exception if other param in kwargs
        if len(kwargs) > 0:
            extra = ','.join('"%s"' % k for k in kwargs.keys())
            raise ValueError('Unrecognized %s' % extra)

        if not hasattr(optimizer, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optimizer, self.update_rule)

        self._reset()

    def _reset(self):
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        self.optim_params ={}
        for p in self.model.params:
            d = {k : v for k,v in self.optim_param.items()}
            self.optim_params[p] = d

    def _step(self):
        mini_batch = sample_coco_minibatch(self.data, batch_size=self.batch_size,
                                           split='train')
        captions, features, urls = mini_batch

        loss, grads = self.model.loss(features, captions)
        self.loss_history.append(loss)

        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_params[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_params[p] = next_config

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        to check accuracy in data: can be train_data or val_data
        :param X:
        :param y:
        :param num_samples:
        :param batch_size: the number of each batch data
        :return:
        """
        N = X.shape[0]

        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            y_pred.append(self.model.sample(X[start:end], max_length=17))

        y_pred = np.vstack(y_pred)
#         print(y_pred.shape)
        acc = np.mean(y_pred == y)
        
        return acc

    def train(self):
        num_train = self.data['train_captions'].shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            if self.verbose and t % self.p_num == 0:
                print('(Iteration %d / %d) loss: % f' % (t + 1, num_iterations, self.loss_history[-1]))

            epoch_end = (t + 1) % iterations_per_epoch == 0
            # 训练完数据一轮
            if epoch_end:
                self.epoch += 1
                for k in self.optim_params:
                    self.optim_params[k]['learning_rate'] *= self.lr_decay

            first_it = (t == 0)
            last_it = (t == num_iterations + 1)
            if first_it or last_it or epoch_end:
                train_captions, train_image_features, _ = sample_coco_minibatch(self.data, batch_size=1000, split='train')
                train_acc = self.check_accuracy(train_image_features, train_captions)
                val_captions, val_image_features, _ = sample_coco_minibatch(self.data, batch_size=1000, split='val')
                val_acc = self.check_accuracy(val_image_features, val_captions)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)

                if self.verbose:
                    print('(Epoch %d / %d), train acc: %f, val_acc: %f)' % (
                    self.epoch, self.num_epochs, train_acc, val_acc))
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        self.model.params = self.best_params