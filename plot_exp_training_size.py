#!/usr/bin/env python3.9

"""Plot algorithms performances depnding on data size"""

import numpy as np
import matplotlib.pyplot as plt

from data_utils import load_data, split_data
from linear_regression import (LinearRegressionMajority,
                               LinearRegressionMedian,
                               LinearRegressionMean,
                               LinearRegressionLeastSquares)
#from exp_hyperparam import learn_all_with_RIDGE, learn_all_with_MP


# Load performance data
npz_ = np.load("exp_training_size_results.npz")
print(npz_)
N = npz_["N"]
train_error_mx = npz_["train_error_mx"]
valid_error_mx = npz_["valid_error_mx"]
learning_time_mx = npz_["learning_time_mx"]
methods_name = npz_["methods_name"]
colors = ["black", "blue", "green", "red"]

print(train_error_mx)
print(N)

# MSE of each method on the validation set depending on dataset size
fig1 = plt.figure()
ax1 = plt.axes()
for i in range(len(methods_name)):
    ax1.loglog(N, valid_error_mx[:, i], color=colors[i],
            linestyle="-", label=methods_name[i])
    if methods_name[i] == "least squares":
        train_error_mx[:, i] = np.round(train_error_mx[:, i], 0) + 1
    ax1.margins(x=0, y=0)
    ax1.set_yticks(np.arange(11) - 1)
    ax1.loglog(N, train_error_mx[:, i], color=colors[i],
        linestyle="dotted", label=methods_name[i])

ax1.set_xlabel("Sample size (N)")
ax1.set_ylabel("Error (MSE)")
ax1.set_title("Performance of different linear regression algorithms"
             + "\n depending on the sample size of the data set")
ax1.legend(loc = "upper right")
plt.savefig("figures/plot_exp_training_size_mse.png")

# time spent by each algo to fit train set depending on the dataset size
fig2 = plt.figure()
ax2 = plt.axes()
for i in range(len(methods_name)):
    ax2.semilogy(N, learning_time_mx[:, i], color=colors[i],
            linestyle="-", label=methods_name[i])

ax2.set_xlabel("Sample size (N)")
ax2.set_ylabel("Time (s)")
ax2.set_title("Time spent by each linear regression algorithm"
             + "to fit the training data set"
             + "\n depending on the sample size of the data set")
ax2.legend(loc = "upper right")
plt.savefig("figures/plot_exp_training_size_time.png")
"""
# Part 2 : learning time of each linear regression method depending on the sample size
fig = plt.figure()
ax = plt.axes()
ax.plot(N, learning_time[:, 0], color="red", linestyle="-", label="majority")
ax.plot(N, learning_time[:, 1], color="blue", linestyle="-", label="median")
ax.plot(N, learning_time[:, 2], color="green", linestyle="-", label="mean")
ax.plot(N, learning_time[:, 3], color="black", linestyle="-", label="least square")
plt.savefig("figures/plot_exp_training_size_time.png")


# Part 10 :
# afficher les performance de prediction de tous les algorithms en fonction de la taille du training set

# load data
X_labeled, y_labeled, X_unlabeled = load_data("data/YearPredictionMSD_100.npz")

X_train_X_valid1, y_train_y_valid1, X_valid2, y_valid2 = split_data(X_labeled, y_labeled, ratio=2/3)

# Ensemble N: tailles differents de S_train
N = np.array([2**i for i in np.arange(6,11+1)])

# Iteration
error_mx = np.zeros((len(N), 6))
for i,n in enumerate(N):
    # reduce size of S_train depending on N
    S_train_X, S_train_y = X_train_X_valid1[0:n, :], y_train_y_valid1[0:n]

    # Ridge
    ridge = learn_all_with_RIDGE(S_train_X, S_train_y)
    ridge.fit(S_train_X, S_train_y)
    y_hat_ridge = ridge.predict(X_valid2)
    lse_ridge = np.mean(np.square(y_hat_ridge - y_valid2))
    
    # Matching Poursuit
    mp = learn_all_with_MP(S_train_X, S_train_y)
    mp.fit(S_train_X, S_train_y)
    y_hat_mp = mp.predict(X_valid2)
    lse_mp = np.mean(np.square(y_hat_mp - y_valid2))
    
    # Orthogonal Matching Poursuit
    # omp = learn_all_with_OMP(S_train_X, S_train_y)
    # omp.fit(S_train_X, S_train_y)
    # y_hat_omp = omp.predict(X_valid2)
    # lse_omp = np.mean(np.square(y_hat_omp - y_valid2))
    lse_omp = 0
    
    # Linear Regression Majority 
    maj = LinearRegressionMajority()
    maj.fit(S_train_X, S_train_y)
    y_hat_maj = maj.predict(X_valid2)
    lse_maj = np.mean(np.square(y_hat_maj - y_valid2))

    # Linear Regression Mean
    mean_ = LinearRegressionMean()
    mean_.fit(S_train_X, S_train_y)
    y_hat_mean = mean_.predict(X_valid2)
    lse_mean = np.mean(np.square(y_hat_mean - y_valid2))

    # Linear Regression Median
    median_ = LinearRegressionMedian()
    median_.fit(S_train_X, S_train_y)
    y_hat_median = median_.predict(X_valid2)
    lse_median = np.mean(np.square(y_hat_median - y_valid2))

    # save results
    error_mx[i, :] = [lse_ridge, lse_mp, lse_omp,
                      lse_maj, lse_mean, lse_median]

print(np.round(error_mx, 0))

fig = plt.figure()
ax = plt.axes()
method = ["ridge", "mp", "omp", "maj", "mean", "median"]
colors = ["b", "g", "r", "c", "m", "k"]
for i in range(len(method)):
    ax.plot(N, error_mx[:, i], color=colors[i], linestyle="-", label=method[i])
plt.legend(loc="upper right")
plt.xlabel("sample size (N)")
plt.ylabel("Least Square Error")
plt.title("Performace of algorithms \n depending on the sample size")
plt.axvline(500, color = 'r', linestyle = ':', label="N=500")    
plt.savefig("figures/plot_exp_training_size.png")


# Part 10.2


def learn_best_predictor_and_predict_test_data(X, y, learn_all_with_ALGO_fct):

    
    # partitionne les donnes etiquetes en un ensemble de taille 500 et un ensemble de validation 2
    S_train_X, S_train_y, X_valid2, y_valid2 = split_data(X, y, ratio=2/3)
    S_train_X, S_train_y = S_train_X[0:n, :], S_train_y[0:n]
    
    # utilise learn_all_with_ALGO()
    # apprend les param w et b a partir de l'ensemble de 500 obs.
    algo = learn_all_with_ALGO_fct(S_train_X, S_train_y)
    algo.fit(S_train_X, S_train_y)
    w_hat = algo.w
    b_hat = algo.b
    y_hat = algo.predict(X_valid2)
    lse_valid2 = np.mean(np.square(y_hat - y_valid2))
    print("\n algorithm performances on sample validation 2 \n" +
          f"\n w = {w_hat} \n b = {b_hat} \n Least Square Error = {lse_valid2} \n")
    acc = np.mean(np.round(y_hat, 0) == np.round(y_valid2, 0))
    print("Accuracy on validation set 2: \n" +
          f"{acc}")
    
    # unlabeled data
    _, _, X_unlabeled = load_data("data/YearPredictionMSD_100.npz")
    y_test = algo.predict(X_unlabeled)
    np.save("test_prediction_results.npy", y_test)
    print(f"\n y_test = {y_test} \n")

# test
X_labeled, y_labeled, _ = load_data("data/YearPredictionMSD_100.npz")
learn_best_predictor_and_predict_test_data(
    X_labeled, y_labeled, learn_all_with_ALGO_fct=learn_all_with_RIDGE)

    
"""

    