#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/16 17:10
@Author  : LI Zhe
"""
from math import ceil, sqrt
import numpy as np
def visualize_grid(Xs, ubound=255.0, padding=1):
    (N, H, W, C)= Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid

def vis_grid(Xs):
    (N, H, W, C)= Xs.shape
    A = int(ceil(sqrt(N)))
    G = np.ones((A * H + A, A * W + A, C), Xs.astype)
    G *= np.min(Xs)
    n = 0
    for y in range(A):
        for x in range(A):
            if n < N:
                G[y * H + y : (y + 1) * H + y, x * W + x : (x + 1) * W + x, :] = Xs[n, :, :, :]
                n += 1
    maxg = G.max()
    ming = G.min()
    G = (G - ming) / (maxg - ming)
    return G

def vis_nn(rows):
    N = len(rows)
    D = len(rows[0])
    H, W, C = rows[0][0].shape
    Xs = rows[0][0]
    G = np.ones((N * H + N, D * W + D, C), Xs.astype)
    for y in range(N):
        for x in range(D):
            G[y * H + y : (y + 1) * H + y, x * W + x : (x + 1) * W + x, :] = rows[x][y]
    maxg = G.max()
    ming = G.min()
    G = (G - ming) / (maxg - ming)
    return G