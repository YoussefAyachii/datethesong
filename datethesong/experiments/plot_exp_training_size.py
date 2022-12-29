#!/usr/bin/env python3.9

"""Plot algorithms performances depnding on data size"""

import numpy as np
import matplotlib.pyplot as plt

from data_utils import load_data, split_data
from linear_regression import (LinearRegressionMajority,
                               LinearRegressionMedian,
                               LinearRegressionMean,
                               LinearRegressionLeastSquares)

from exp_hyperparam import learn_all_with_RIDGE
from exp_hyperparam import learn_all_with_MP
from exp_hyperparam import learn_all_with_OMP


# Load algo performances' data
npz_ = np.load("exp_training_size_results.npz")
N = npz_["N"]
train_error_mx = npz_["train_error_mx"]
valid_error_mx = npz_["valid_error_mx"]
learning_time_mx = npz_["learning_time_mx"]
methods_name = npz_["methods_name"]
colors = ["black", "blue", "green", "red"]


# MSE of each method on the validation set depending on dataset size
fig1 = plt.figure()
ax1 = plt.axes()
for i in range(len(methods_name)):
    ax1.loglog(N, valid_error_mx[:, i], color=colors[i],
            linestyle="-", label=f"{methods_name[i]} (valid)")
    if methods_name[i] == "least squares":
        # trick to see the dashed line on the x axis when low N
        train_error_mx[:, i] = np.round(train_error_mx[:, i], 0) + 1
    ax1.margins(x=0, y=0)
    ax1.set_yticks(np.arange(11) - 1)
    ax1.loglog(N, train_error_mx[:, i], color=colors[i],
        linestyle="dotted", label=f"{methods_name[i]} (train)")

ax1.set_xlabel("Sample size (N)")
ax1.set_ylabel("Loss function (MSE)")
ax1.set_title("Performance of different linear regression algorithms"
              + "\n when fiting the training set and the validation set"
              + "\n according to the sample size")
ax1.legend(loc = "upper right")
plt.tight_layout()
plt.savefig("figures/plot_exp_training_and_validation_size_mse.png")

# time spent by each algo to fit train set depending on the dataset size
fig2 = plt.figure()
ax2 = plt.axes()
for i in range(len(methods_name)):
    ax2.semilogy(N, learning_time_mx[:, i], color=colors[i],
            linestyle="-", label=methods_name[i])

ax2.set_xlabel("Sample size (N)")
ax2.set_ylabel("Time (s)")
ax2.set_title("Time spent by each linear regression algorithm"
             + " to fit the training data set"
             + "\n according to the sample size")
ax2.legend(loc = "upper right")
plt.savefig("figures/plot_exp_training_and_validation_size_time.png")


# 10.
# display performances of all algorithms according to sample size

# load data
X_labeled, y_labeled, X_unlabeled = load_data(
    "data/YearPredictionMSD_100.npz")

X_train_X_valid1, y_train_y_valid1, X_valid2, y_valid2 = split_data(
    X_labeled, y_labeled, ratio=2/3)

# Ensemble N: tailles differents de S_train
N = np.array([2**i for i in np.arange(6,11+1)])

# Iteration
error_mx = np.zeros((len(N), 7))
for i,n in enumerate(N):
    # reduce size of S_train depending on N
    S_train_X, S_train_y = X_train_X_valid1[0:n, :], y_train_y_valid1[0:n]

    # Ridge
    ridge = learn_all_with_RIDGE(S_train_X, S_train_y)
    ridge.fit(S_train_X, S_train_y)
    y_hat_ridge = ridge.predict(X_valid2)
    mse_ridge = np.mean(np.square(y_hat_ridge - y_valid2))
    
    # Matching Poursuit
    mp = learn_all_with_MP(S_train_X, S_train_y)
    mp.fit(S_train_X, S_train_y)
    y_hat_mp = mp.predict(X_valid2)
    mse_mp = np.mean(np.square(y_hat_mp - y_valid2))
    
    # Orthogonal Matching Poursuit
    omp = learn_all_with_OMP(S_train_X, S_train_y)
    omp.fit(S_train_X, S_train_y)
    y_hat_omp = omp.predict(X_valid2)
    mse_omp = np.mean(np.square(y_hat_omp - y_valid2))
    
    # Linear Regression Majority 
    maj = LinearRegressionMajority()
    maj.fit(S_train_X, S_train_y)
    y_hat_maj = maj.predict(X_valid2)
    mse_maj = np.mean(np.square(y_hat_maj - y_valid2))

    # Linear Regression Mean
    mean_ = LinearRegressionMean()
    mean_.fit(S_train_X, S_train_y)
    y_hat_mean = mean_.predict(X_valid2)
    mse_mean = np.mean(np.square(y_hat_mean - y_valid2))

    # Linear Regression Median
    median_ = LinearRegressionMedian()
    median_.fit(S_train_X, S_train_y)
    y_hat_median = median_.predict(X_valid2)
    mse_median = np.mean(np.square(y_hat_median - y_valid2))

    # Linear Regression Least Squares 
    ls = LinearRegressionLeastSquares()
    ls.fit(S_train_X, S_train_y)
    y_hat_ls = ls.predict(X_valid2)
    mse_ls = np.mean(np.square(y_hat_ls - y_valid2))

    # save results
    error_mx[i, :] = [mse_ridge, mse_mp, mse_omp, mse_maj,
                      mse_mean, mse_median, mse_ls]

fig = plt.figure()
ax = plt.axes()
method = ["ridge", "mp", "omp", "maj", "mean", "median", "ls"]
colors = ["b", "g", "r", "c", "m", "k", "y"]
for i in range(len(method)):
    ax.loglog(N, error_mx[:, i], color=colors[i], linestyle="-", label=method[i])
plt.legend(loc="upper right")
plt.xlabel("Sample size (N)")
plt.ylabel("Loss function (MSE)")
plt.title("Performace of algorithms when predicting the valid2 set"
          + "\n depending on the sample size")
plt.axvline(500, color = 'r', linestyle = ':', label="N=500")    
plt.savefig("figures/plot_exp_training_size.png")


# Part 10.2


def learn_best_predictor_and_predict_test_data(
    X, y, learn_all_with_ALGO, X_unlabeled, file_path):
    """Learn the best model and predict the unlabeled test data.
    The result is automatically saved as an .npy file
    
    -------
    Parameters
    X: nd.array[n, d]
        Whole training data set
    y: nd.array[n]
        Labels of the training data set
    learn_all_with_ALGO: function
        learn_all_with_ALGO function to use depending on
        the chosen algorithm
    X_unlabeled: nd.array[n2, d2]
        Unlabeled data set that want to labelise
    file_path: str
        File name, path and file format in a unique str.
    -------
    Returns
    .npy: file
        Results of algorithms performances saved into a .npy file.
    """
    
    # partitionne les donnes etiquetes en train (n=500) et valid2
    S_train_X, S_train_y, X_valid2, y_valid2 = split_data(X, y, ratio=2/3)
    n = 500
    S_train_X, S_train_y = S_train_X[0:n, :], S_train_y[0:n]
    
    # utilise learn_all_with_ALGO()
    # apprend les param w et b a partir de l'ensemble train
    algo = learn_all_with_ALGO(S_train_X, S_train_y)
    algo.fit(S_train_X, S_train_y)
    w_hat = algo.w
    b_hat = algo.b
    y_hat = algo.predict(X_valid2)
    mse_valid2 = np.mean(np.square(y_hat - y_valid2))
    print("\n algorithm performances on sample validation 2 \n" +
          f"\n w = {w_hat} \n b = {b_hat} \n"
          + f"Loss Function (MSE) valid2 set = {mse_valid2} \n")

    # unlabeled data
    y_test = algo.predict(X_unlabeled)
    np.save(file_path, y_test)

# test
learn_best_predictor_and_predict_test_data(
    X=X_labeled, y=y_labeled,
    learn_all_with_ALGO=learn_all_with_RIDGE,
    X_unlabeled=X_unlabeled,
    file_path="test_prediction_results.npy")
