#!/usr/bin/env python3.9

"""testing manually different values of ridge regression coefficient"""

import numpy as np
import matplotlib.pyplot as plt

from datethesong.algorithms.data_utils import(
    load_data, randomize_data, split_data)
from datethesong.algorithms.linear_regression import LinearRegressionRidge


# fixer arbitrairement le parametre kmax ou lamda
lambda_ridge = 0.02

# charger les donnees etiquetees et en selectioner un sous ensemble
X_labeled, y_labeled, X_unlabeled = load_data(
    "datethesong/data/YearPredictionMSD_100.npz")
N = 500
X_labeled = X_labeled[:500, :N]
y_labeled = y_labeled[:N]
X_unlabeled = X_unlabeled[:N]

X_train, y_train, X_test, y_test = split_data(
    X_labeled, y_labeled, ratio=2/3)

# Learning
ridge_reg = LinearRegressionRidge(lambda_ridge=lambda_ridge)
ridge_reg.fit(X_train, y_train)

y_hat = ridge_reg.predict(X_test)
errors = np.zeros(len(y_test))
mse_evolutive = np.zeros(len(y_test))
for i in range(len(y_test)):
    resid_error = abs(y_hat[i] - y_test[i])
    errors[i] = resid_error
    if i>0:
        mse_evolutive[i] = np.mean(np.square(errors[:i]))
    else:
        mse_evolutive[i] = errors[0]


fig = plt.figure()
ax = plt.axes()
plt.plot(mse_evolutive)
plt.title("Progressive Mean Square Error of Ridge Regression method \n"
          + "for a user fixed hyperparameter lambda")
plt.xlabel("Iteration number")
plt.ylabel("Error (MSE)")
plt.savefig("figures/exp_ridge_mse_evolutive.png")
