import random
import numpy as np

def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    a naive implementation of numericak gradient of f at x
    :param f: a function that takes a single argument
    :param x: the point (at 1-D array) to evaluate the gradient
    :param verbose:  to control showing the gradient or not
    :param h: distance near the x point
    :return:
    """
    # evaluate function value at original point
    # fx = f(x)
    grad = np.zeros_like(x)

    # np.nditer is the iterator of numpy array
    # flags = ['multi_index'] 对 a 进行多重索引
    # op_flags = ['readwrite'] ：对 a 可以进行读取和写入
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        ix = it.multi_index # 根据x中的元素个数迭代多少次
        old_val = x[ix]
        x[ix] = old_val + h
        fxph = f(x)
        x[ix] = old_val - h
        fxmh = f(x)
        x[ix] = old_val
        grad[ix] = (fxph - fxmh) / (2 * h)
        if verbose:
            print(ix, grad[ix])
        it.iternext()

    return grad

def eval_numerical_gradient_array(f, x, df,verbose=False, h=0.00001):
    """
    a naive implementation of numericak gradient of f at numpy x
    :param f: a function that takes a single argument
    :param x: array  to evaluate the gradient
    :param verbose:  to control showing the gradient or not
    :param h: distance near the x point
    :return:
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        ix = it.multi_index # 根据x中的元素个数迭代多少次

        old_val = x[ix] # 一次取出x的每一行
        x[ix] = old_val + h # boardcasting
        fxph = f(x).copy() # the old list remains unchanged even when the new list is modified.
        x[ix] = old_val - h
        fxmh = f(x).copy()
        x[ix] = old_val
        grad[ix] = np.sum((fxph - fxmh) * df) / (2 * h) # 最后保留x[0]的维度
        if verbose:
            print(ix, grad[ix])
        it.iternext()

    return grad

def gradient_check(f, x, analytic_grad, num_checks=10, h=1e-5):

    for i in range(num_checks):
        ix = tuple([random.randrange(m) for m in x.shape])
        oldval = x[ix]
        x[ix] = oldval + h
        fxph = f(x)
        x[ix] = oldval - h
        fxmh = f(x)
        x[ix] = oldval

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]

        relative_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        print('numerical grad: %f, analytic grad: %f, the ralative error is %e' % (grad_numerical, grad_analytic, relative_error))