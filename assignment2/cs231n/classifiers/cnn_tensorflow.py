#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/2 23:15
@Author  : LI Zhe
"""
import numpy as np
import tensorflow as tf
# import classifiers.functions as F
import cs231n.classifiers.functions as F
from sklearn.metrics import accuracy_score


class BaseCNNClassifier(object):
    def __init__(self, image_size=24, num_classes=10, batch_size=50, channels=3):
        self._image_size = image_size
        self._num_classes = num_classes
        self._batch_size = batch_size
        self._channels = channels
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self._session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
         
        self._images = tf.placeholder(tf.float32, shape=[None,self._channels, self._image_size, self._image_size])
        self._labels = tf.placeholder(tf.int64, shape=[None])
        self._keep_prob = tf.placeholder(tf.float32)
        
        self._global_step = tf.Variable(0, tf.int64, name="global_step")
        self._logits = self._inference(self._images, self._keep_prob)
        self._avg_loss = self._loss(self._labels, self._logits)
        self._train_op = self._train(self._avg_loss)
        self._accuracy = F.accuracy_score(self._labels, self._logits)
        self._saver = tf.train.Saver(tf.global_variables())
        self._session.run(tf.global_variables_initializer())

    def fit(self, X, y, max_epoch=10):
        for epoch in range(max_epoch):
            for i in range(0, len(X), self._batch_size):
                batch_images, batch_labels = X[i:i + self._batch_size], y[i:i + self._batch_size]
                feed_dict = {self._images: batch_images, self._labels: batch_labels, self._keep_prob: 0.5}
                _, train_avg_loss, global_step = self._session.run(
                    fetches=[self._train_op, self._avg_loss, self._global_step], feed_dict=feed_dict)
            print("epochs =", global_step)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        res = None
        for i in range(0, len(X), self._batch_size):
            batch_images = X[i:i + self._batch_size]
            feed_dict = {self._images: batch_images, self._keep_prob: 1.0}
            test_logits = self._session.run(fetches=self._logits, feed_dict=feed_dict)
            if res is None:
                res = test_logits
            else:
                res = np.r_[res, test_logits]
        return res

    def score(self, X, y):
        total_acc, total_loss = 0, 0
        for i in range(0, len(X), self._batch_size):
            batch_images, batch_labels = X[i:i + self._batch_size], y[i:i + self._batch_size]
            feed_dict = {self._images: batch_images, self._labels: batch_labels, self._keep_prob: 1.0}
            acc, avg_loss = self._session.run(fetches=[self._accuracy, self._avg_loss], feed_dict=feed_dict)
            total_acc += acc * len(batch_images)
            total_loss += avg_loss * len(batch_images)
        return total_acc / len(X), total_loss / len(X)

    def save(self, filepath):
        self._saver.save(self._session, filepath)

    def _inference(self, X, keep_prob):
        pass

    def _loss(self, labels, logits):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        tf.add_to_collection('losses', cross_entropy_mean)
        return tf.add_n(tf.get_collection('losses'))

    def _train(self, avg_loss):
        return tf.train.AdamOptimizer().minimize(avg_loss, self._global_step)


class CNN_tensorflow(BaseCNNClassifier):
    def _inference(self, X, keep_prob):
        h = F.max_pool(F.activation(F.conv(X, 64)))
        h = F.max_pool(F.activation(F.conv(h, 128)))
        h = F.max_pool(F.activation(F.conv(h, 256)))
        h = F.activation(F.dense(F.flatten(h), 1024))
        h = F.dense(h, self._num_classes)
        return h
    
class CNN_tensorflow_2(BaseCNNClassifier):
    def _inference(self, X, keep_prob):
        h = F.max_pool(F.activation(F.conv(X, 64)))
        h = F.max_pool(F.activation(F.conv(h, 128)))
        h = F.max_pool(F.activation(F.conv(h, 256)))
        h = F.activation(F.dense(F.flatten(h), 1024))
        h = tf.nn.dropout(h, keep_prob)
        h = F.dense(h, self._num_classes)
        return h
    
class CNN_tensorflow_3(BaseCNNClassifier):
    def _inference(self, X, keep_prob):
        h = F.max_pool(F.activation(F.conv(X, 64)))
        h = F.activation(F.conv(X, 64))
        h = F.max_pool(F.activation(F.conv(h, 128)))
        h = F.max_pool(F.activation(F.conv(h, 256)))
        h = F.activation(F.dense(F.flatten(h), 1024))
        h = tf.nn.dropout(h, keep_prob)
        h = F.dense(h, self._num_classes)
        return h
class CNN_tensorflow_4(BaseCNNClassifier):
    def _inference(self, X, keep_prob):
        h = F.max_pool(F.activation(F.conv(X, 64)))
        h = F.activation(F.conv(X, 64))
        h = F.activation(F.conv(X, 64))
        h = F.max_pool(F.activation(F.conv(h, 128)))
        h = F.max_pool(F.activation(F.conv(h, 256)))
        h = F.activation(F.dense(F.flatten(h), 1024))
        h = tf.nn.dropout(h, keep_prob)
        h = F.dense(h, self._num_classes)
        return h

class CNN_tensorflow_5(BaseCNNClassifier):
    def _inference(self, X, keep_prob):
        h = F.max_pool(F.activation(F.conv(X, 64)))
        h = F.activation(F.conv(h, 64))
        h = F.max_pool(F.activation(F.conv(h, 128)))
        h = F.activation(F.conv(h, 128))
        h = F.max_pool(F.activation(F.conv(h, 256)))
        h = F.activation(F.dense(F.flatten(h), 1024))
        h = tf.nn.dropout(h, keep_prob)
        h = F.dense(h, self._num_classes)
        return h

class CNN_tensorflow_6(BaseCNNClassifier):
    def _inference(self, X, keep_prob):
        h = F.max_pool(F.activation(F.conv(X, 64)))
        h = F.activation(F.conv(h, 64))
        h = F.max_pool(F.activation(F.conv(h, 128)))
        h = F.activation(F.conv(h, 128))
        h = F.max_pool(F.activation(F.conv(h, 256)))
        h = F.activation(F.dense(F.flatten(h), 1024))
        h = tf.nn.dropout(h, keep_prob)
        h = F.activation(F.dense(h, 256))
        h = tf.nn.dropout(h, keep_prob)
        h = F.dense(h, self._num_classes)
        return h