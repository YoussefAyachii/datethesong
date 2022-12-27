#!/usr/bin/env python3.9

"""Compare performances of linear regression algorithms
depending on the size of the training dataset
"""

import numpy as np
from time import time

from data_utils import load_data, split_data
from linear_regression import (LinearRegressionMean,
                               LinearRegressionMedian,
                               LinearRegressionMajority,
                               LinearRegressionLeastSquares)


def algo_perf_dep_on_dataset_size(X_train, y_train,
                                  X_valid, y_valid,
                                  method=LinearRegressionLeastSquares(),
                                  N=None):
    """Mesure performances of linear regression algorithms
    depending on the size of the training dataset.

    Parameters
    ----------
    X_train : np.ndarray [n1, d1]
        Training set feature array of n feature vectors with size d
    y_train : np.ndarray [n1]
        Training set label vector of dimension n
    X_valid : np.ndarray [n2, d2]
        Validation set array of n feature vectors with size d
    y_valid : np.ndarray [n2]
        Validation set label vector of dimension n
    method : class
        Class of the chosen algorithm
    N : np.ndarray [N]
        Array of different sizes to test the algo on the data set.

    Returns
    -------
    train_error : np.ndarray [N]
        Vector of the prediction error of the algo on the different
        sizes of the dataset on the training set
    valid_error : np.ndarray [N]
        Vector of the prediction error of the algo on the different
        sizes of the dataset on the validation set
    learning_time :
        Vector of the time spent by the algorithm to learn its parameters.
    """
    assert N.ndim == 1
    assert N.shape[0] > 1
    
    train_error = np.zeros(len(N))
    valid_error = np.zeros(len(N))
    learning_time = np.zeros(len(N))

    for i, n in enumerate(N):
        X_train_temp, y_train_temp = X_train[0:n, :], y_train[0:n]
        
        # Linear Regression Majority 
        algo = method
        t0 = time()
        algo.fit(X_train_temp, y_train_temp)
        t1 = time()
        
        # prediction error on validation set
        yhat_valid = algo.predict(X_valid)
        valid_error[i] = np.mean(np.square(yhat_valid - y_valid)) ** 2
        
        # prediction error on training set
        yhat_train = algo.predict(X_train_temp)
        train_error[i] = np.mean(np.square(yhat_train - y_train_temp)) ** 2

        # learning time
        learning_time[i] = t1 - t0

    return train_error, valid_error, learning_time


# load data and fix data sizes vector N
X_labeled, y_labeled, _ = load_data(
    "data/YearPredictionMSD_100.npz")

S_train_X, S_train_y, S_valid_X, S_valid_y = split_data(
    X_labeled, y_labeled, 2/3)

N = np.array([2**i for i in np.arange(5, 11 + 1)])

# apply function on all methods and save results
methods = (LinearRegressionMean(),
           LinearRegressionMedian(),
           LinearRegressionMajority(),
           LinearRegressionLeastSquares())
methods_name = ["mean", "median", "majority", "least squares"]

train_error_mx = np.zeros((len(N), len(methods)))
valid_error_mx = np.zeros((len(N), len(methods)))
learning_time_mx = np.zeros((len(N), len(methods)))

for i in range(len(methods)):
    method_temp = methods[i]
    train_error, valid_error, learning_time = algo_perf_dep_on_dataset_size(
        S_train_X, S_train_y, S_valid_X, S_valid_y,
        method=methods[i], N=N)
    train_error_mx[:, i] = train_error
    valid_error_mx[:, i] = valid_error
    learning_time_mx[:, i] = learning_time

np.savez("exp_training_size_results.npz",
    N=N,
    methods_name=methods_name,
    train_error_mx=train_error_mx,
    valid_error_mx=valid_error_mx,
    learning_time_mx=learning_time_mx)

print("\n valid error: \n ", valid_error_mx)
print("\n train error: \n ", train_error_mx)
print("\n N vector: \n", N)
print("\n learning_time: \n", learning_time_mx)
