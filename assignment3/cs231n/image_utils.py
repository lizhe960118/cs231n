#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/7 12:15
@Author  : LI Zhe
"""
import os, tempfile
from urllib import request
from urllib import error
import numpy as np
from scipy.misc import imread
from cs231n.layer_faster import conv_forward_fast
# from layer_faster import conv_forward_fast

def blur_image(X):
    w_blur = np.zeros((3, 3, 3, 3))
    b_blur = np.zeros(3)
    blur_param = {'stride':1, 'pad':1}
    for i in range(3):
        w_blur[i, i] = np.asarray([[1, 2, 1], [2, 188, 2], [1, 2, 1]], dtype=np.float32)
    w_blur /= 200
    return conv_forward_fast(X, w_blur, b_blur, blur_param)[0]

def preprocess_image(img, mean_img, mean='image'):
    """
    Convert to float. transpose, and subtract mean pixel
    :param img:(H, W, 3)
    :param mean_img:
    :param mean:
    :return:
        - (1, 3, H, 3)
    """
    if mean == 'image':
        mean = mean_img
    elif mean ==  'pixel':
        mean = mean_img.mean(axis= (1, 2), keepdims=True)
    elif mean == 'none':
        mean = 0
    else:
        raise ValueError('mean must be image or pixel or none')
    return img.astype(np.float32).transpose(2, 0, 1)[None] - mean

def deprocess_image(img, mean_img, mean='image', renorm=False):
    """
    Add mean pixel. transpose, and convert to uint8
    :param img:
    :param mean_img:
    :param mean:
    :param renorm:
    :return:
        - (H, W, 3)
    """
    if mean == 'image':
        mean = mean_img
    elif mean ==  'pixel':
        mean = mean_img.mean(axis = (1, 2), keepdims=True)
    elif mean == 'none':
        mean = 0
    else:
        raise ValueError('mean must be image or pixel or none')
    if img.ndim == 3:
        img = img[None]
    img = (img + mean).transpose(1, 2, 0)
    if renorm:
        low, high = img.min(), img.max()
        img = 255.0 * (img - low) / (high - low)
    return img.astype(np.uint8)

def image_from_url(url):
    try:
        f = request.urlopen(url)
        _, fname = tempfile.mkstemp()
        with open(fname, 'wb') as ff:
            ff.write(f.read())
        img = imread(fname)
        os.remove(fname)
        return img
    except error.URLError as e:
        print('URL Error: ', e.reason, url)
    except error.HTTPError as e:
        print('HTTP Error:', e.code, url)


