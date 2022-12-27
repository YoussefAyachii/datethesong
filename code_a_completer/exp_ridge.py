#!/usr/bin/env python3.9

"""testing manually different values of ridge regression coefficient"""

import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_data, randomize_data, split_data
from linear_regression import LinearRegressionRidge


# fixer arbitrairement le parametre kmax ou lamda
lambda_ridge = 0.02

# charger les donnees etiquetees et en selectioner un sous ensemble
X_labeled, y_labeled, X_unlabeled = load_data(
    "data/YearPredictionMSD_100.npz")
N = 500
X_labeled = X_labeled[:500, :N]
y_labeled = y_labeled[:N]
X_unlabeled = X_unlabeled[:N]

X_train, y_train, X_test, y_test = split_data(
    X_labeled, y_labeled, ratio=2/3)

# Learning
ridge_reg = LinearRegressionRidge(lambda_ridge=lambda_ridge)
ridge_reg.fit(X_train, y_train)

# controler le comportement de votre algo en affichant (plot) et commentant l'erreur du residu en en fonction des iteration
# (valeurs initiales et finales, decroissance, coherence avec les coefficients de regression obtenus, nmbre d'iterations)
y_hat = ridge_reg.predict(X_test)
errors = np.zeros(len(y_test))
lse_evolutive = np.zeros(len(y_test))
for i in range(len(y_test)):
    error = np.square(y_hat[i] - y_test[i])
    
    errors[i] = error
    lse_evolutive[i] = np.mean(np.sum(errors[:i]))

fig = plt.figure()
ax = plt.axes()
plt.plot(lse_evolutive)
plt.show()

least_square_error = np.mean(np.sum(np.square(y_hat - y_test)))
print(least_square_error)