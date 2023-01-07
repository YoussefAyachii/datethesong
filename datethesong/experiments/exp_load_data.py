#!/usr/bin/env python3.9

"""Exploring data set"""

import numpy as np

from datethesong.algorithms.data_utils import load_data


X_labeled, y_labeled, X_unlabeled = load_data(
    "datethesong/data/YearPredictionMSD_100.npz")

# infos on X_labeled, y_labeled and X_unlabeled
print("\n X_labeled: \n type: {} \n dim: {} \n"
      + "shape: {} \n data type: {}"
      .format(type(X_labeled), X_labeled.ndim,
              X_labeled.shape, X_labeled.dtype))
print("\n y_labeled: \n type: {} \n dim: {} \n"
      + "shape: {} \n data type: {}"
      .format(type(y_labeled), y_labeled.ndim,
              y_labeled.shape, y_labeled.dtype))
print("\n X_unlabeled: \n type: {} \n dim: {} \n"
      + "shape: {} \n data type: {}"
      .format(type(X_unlabeled), X_unlabeled.ndim,
              X_unlabeled.shape, X_unlabeled.dtype))

# first 5 values of y_labeled
print("\n first 5 values of y_labeled: \n", y_labeled[:5])

# last 5 values of y_labeled:
print("\n last 5 values of y_labeled: \n", y_labeled[-5:])

# 2 premiers coefficient de la première et la dernière ligne
print("\n 2 premiers coefficients de la première ligne"
      + "\n et de la dernière ligne de X_labeled: \n {}"
      .format(X_labeled[[0,-1],0:2]))
print("\n 2 premiers coefficients de la première ligne"
      + "\n et de la dernière ligne de X_unlabeled: \n {}"
      .format(X_unlabeled[[0,-1],0:2]))

# le dernier coefficient de la première et la dernière ligne
print("\n le dernier coefficient de la première"
      + "\n et la dernière ligne de X_labeled: \n {}"
      .format(X_labeled[[0,-1],-1]))
print("\n le dernier coefficient de la première"
      + "\n et la dernière ligne de X_unlabeled: \n {}"
      .format(X_unlabeled[[0,-1],-1]))
