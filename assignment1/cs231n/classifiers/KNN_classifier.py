import numpy as np


class KNearestNeighbor(object):
    """docstring NearestNeighbor"""

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Inputs:
        - X(X_train): A numpy array of shape (num_train, D) containing training data
          consisting of num_train samples each of dimension D
        """
        self.X_train = X
        self.y_train = y

    def compute_distances_two_loops(self, X):
        """
        Inputs:
        - X(X_test): A numpy array of shape (num_test, D) containing test data
        Returns:
        - dist: A numpy array of shape (num_test, num_train) where dists[i,j] is
          the Euclidean distances between the ith test point and the jth training point
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i, j] = np.sqrt(
                    np.sum(np.square(X[i] - self.X_train[j])))
        #         Compute the L2 distance
        return dists

    def compute_distances_one_loop(self, X):
        """
        compute the distance between each test data and whole train data
        :param X: X_train
        :return: dists: A numpy array of shape (num_test, num_train)
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            # using the broadcasting
            dists[i] = np.sqrt(np.sum(np.square(X[i] - self.X_train), axis=1))
        return dists

    def compute_distances_zero_loop(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        # (x-y)^2 = x^2 + y^2 - 2*xy
        dists = np.sqrt(np.sum(np.square(self.X_train), axis=1) +
                        np.transpose([np.sum(np.square(X), axis=1)]) -
                        2 *
                        np.dot(X, self.X_train.T))
        return dists

    def predict_label(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # closest_y = []
            # np.argsort()可以对dist进行排序选出k个最近的训练样本
            # np.bincount()会统计输入数组出现的频数
            # 结合np.argmax()就可以实现vote机制。
            closest_y = self.y_train[np.argsort(dists[i])[:k]]
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred

    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances_zero_loop(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)
        return self.predict_label(dists, k=k)
