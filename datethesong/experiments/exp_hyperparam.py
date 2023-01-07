#!/usr/bin/env python3.9

"""Chosing best hyperparameter"""


import numpy as np
import matplotlib.pyplot as plt

from datethesong.algorithms.data_utils import(
    load_data, split_data)
from datethesong.algorithms.linear_regression import(
    LinearRegressionRidge, LinearRegressionMp, LinearRegressionOmp)

# load data
X_labeled, y_labeled, X_unlabeled = load_data(
    "datethesong/data/YearPredictionMSD_100.npz")
X_train, y_train, X_valid, y_valid = split_data(
    X_labeled, y_labeled, ratio=2/3)

N = 500
X_train, y_train = X_train[:N, :], y_train[:N]

def best_hyper(hypers, method, X_train,
               y_train, X_valid, y_valid):
    """Determine the best hyperparameter for the algo
    among a set of hyperparameter values.

    ---------
    Parameters
    hypers: nd.array[n]
        Hyperparameter values to test
    method: class
        Algorithm on for which you want to chose the best
        hyerparameter value
    X_train: ndarray[n1, d1]
        Training data set.
    y_train: ndarray[n1]
        Labels of the training data set
    X_valid:  ndarray[n2, d2]
        Validation set
    y_valid: ndarray[n2]
        Labels of the validation data set

    --------
    Returns:
    best: float
        Hyperparameter value that minimizes best the loss function

    """
    lse_train = np.zeros(len(hypers))
    lse_valid = np.zeros(len(hypers))
    results = np.zeros((len(hypers), 3))
    for i, h in enumerate(hypers):
        algo = method(h)
        algo.fit(X_train, y_train)

        y_hat_train = algo.predict(X_train)
        lse_train[i] = np.linalg.norm(y_hat_train - y_train, ord=2)

        y_hat_valid = algo.predict(X_valid)
        lse_valid[i] = np.linalg.norm(y_hat_valid - y_valid, ord=2)

        results[i, :] = np.array([h, lse_train[i], lse_valid[i]])
    return results

# trial
lambdas = np.logspace(0.0001, 1, num=50)/10 
results_ridge = best_hyper(hypers=lambdas,
                           method=LinearRegressionRidge,
                           X_train=X_train,
                           y_train=y_train,
                           X_valid=X_valid,
                           y_valid=y_valid)


def hyper_plot(hypers, hyper_name, method,
               method_name, file_path, X_train,
               y_train, X_valid, y_valid):
    """Plot the best_hyper() results.
    Plots the LSE values for both the training set and
    the validation set depending of the hyperarameter values

    ---------
    Parameters
    hypers: nd.array[n]
        Hyperparameter values to test.
    hyper_name: str
        hyperparameter name (lambda or kmax).
    method: class
        Algorithm on for which you want to chose the best
        hyerparameter value.
    method name: str
        Name of the method to use in the plot title.
    file_path: str
        Path to save the plot in.
        Must include the filename and the extension (i.g. .png)
    X_train: ndarray[n1, d1]
        Training data set.
    y_train: ndarray[n1]
        Labels of the training data set
    X_valid:  ndarray[n2, d2]
        Validation set
    y_valid: ndarray[n2]
        Labels of the validation data set
    
    ----------
    Retruns:
    plot: .png file
        generate and save the resulting plot (see function
        description for more information).
    """
    results = best_hyper(hypers,
                         method,
                         X_train=X_train,
                         y_train=y_train,
                         X_valid=X_valid,
                         y_valid=y_valid)
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(results[:, 0], results[:, 1], label="training set",
            linestyle="-", color="green")
    ax.plot(results[:, 0], results[:, 2], label="validation set",
            linestyle=":", color="blue")
    ax.scatter(results[np.argmin(results[:, 2]), 0],
               np.min(results[:, 2]), color="red",
               label="best lambda = {}\nlowest mse = {}"
               .format(round(results[np.argmin(results[:, 2]), 0], 3),
                       round(min(results[:, 2]), 3)))
    plt.xlabel("{}".format(hyper_name))
    plt.ylabel("Least Square Error")
    plt.legend(loc="upper right")
    plt.title(f"{method_name} : \n Least square error values "
              + f"for different values of {hyper_name} ")
    plt.savefig(file_path)


# Ridge regression
lambdas = np.logspace(0.0001, 1, num=50)/10
results_ridge = hyper_plot(hypers=lambdas,
                           hyper_name="lambda",
                           method=LinearRegressionRidge,
                           method_name="ridge regression",
                           file_path="figures/hyperparam_ridge.png",
                           X_train=X_train,
                           y_train=y_train,
                           X_valid=X_valid,
                           y_valid=y_valid)

# Matching Poursuit
kmax = np.arange(200)
results_mp = hyper_plot(hypers=kmax,
                        hyper_name="kmax",
                        method=LinearRegressionMp,
                        method_name="matching poursuit",
                        file_path="figures/hyperparam_mp.png",
                        X_train=X_train,
                        y_train=y_train,
                        X_valid=X_valid,
                        y_valid=y_valid)


# Orthogonal Matching Poursuit
kmax = np.arange(25)
results_mp = hyper_plot(hypers=kmax,
                        hyper_name="kmax",
                        method=LinearRegressionOmp,
                        method_name="orthogonal matching poursuit",
                        file_path="figures/hyperparam_omp.png",
                        X_train=X_train,
                        y_train=y_train,
                        X_valid=X_valid,
                        y_valid=y_valid)


# 9. Estimation des hyperparametres par validation croisee

def learn_all_with_RIDGE(X,y):
    """Determine the best ridge hyper parameter (lambda)

    --------
    Parameters
    X: ndarray[n, d]
        Training data set.
    y_train: ndarray[n]
        Labels of the training data set

    --------
    Returns
    best_lambda: float
        return LinearRegressionRidge object with already defined
        lambda_ridge arg.
    """
    X_train, y_train, X_valid1, y_valid1 = split_data(X, y, ratio=2/3)

    # ensemble de valeur a tester pour l'hyperparametre lambda
    lambdas = np.logspace(0.0001, 1, num=20)/10 

    # appliquer l'algorithme
    results = np.zeros((len(lambdas), 3))
    for i, l in enumerate(lambdas):
        ridge_reg = LinearRegressionRidge(lambda_ridge=l)
        ridge_reg.fit(X_train, y_train)

        # error on learning set
        y_hat_train = ridge_reg.predict(X_train)
        lse_train = np.mean(np.square(y_hat_train - y_train))

        # error on validation set
        y_hat_valid1 = ridge_reg.predict(X_valid1)
        lse_valid1 = np.mean(np.square(y_hat_valid1 - y_valid1))
        
        # save results
        results[i,:] = np.array([l, lse_train, lse_valid1])

    # best value of hyperparameter lambda
    best_lambda = results[np.argmin(results[:, 2]), 0]
        
    return LinearRegressionRidge(lambda_ridge=best_lambda)


def learn_all_with_MP(X,y):
    """Determine the best ridge hyper parameter (kmax)
    for the matching poursuit algorithm.

    --------
    Parameters
    X: ndarray[n, d]
        Training data set.
    y_train: ndarray[n]
        Labels of the training data set

    --------
    Returns
    best_lambda: float
        return LinearRegressionMp object with already defined
        kmax arg.
    """

    X_train, y_train, X_valid1, y_valid1 = split_data(X, y, ratio=2/3)

    # ensemble de valeur a tester pour l'hyperparametre kmax
    kmax = np.arange(200)

    # appliquer l'algorithme
    results = np.zeros((len(kmax), 3))
    for i, k in enumerate(kmax):
        mp = LinearRegressionMp(n_iter=k)
        mp.fit(X_train, y_train)

        # error on learning set
        y_hat_train = mp.predict(X_train)
        lse_train = np.linalg.norm(y_hat_train - y_train, ord=2)

        # error on validation set
        y_hat_valid1 = mp.predict(X_valid1)
        lse_valid1 = np.linalg.norm(y_hat_valid1 - y_valid1, ord=2)

        # save results
        results[i,:] = np.array([k, lse_train, lse_valid1])

    # best value of hyperparameter lambda
    best_kmax = int(results[np.argmin(results[:, 2]), 0])
    
    return LinearRegressionMp(n_iter=best_kmax)


def learn_all_with_OMP(X,y):
    """Determine the best ridge hyper parameter (kmax)
    for the orthogonal matching poursuit algorithm.

    --------
    Parameters
    X: ndarray[n, d]
        Training data set.
    y_train: ndarray[n]
        Labels of the training data set

    --------
    Returns
    best_lambda: float
        return LinearRegressionOmp object with already defined
        kmax arg.
    """

    X_train, y_train, X_valid1, y_valid1 = split_data(X, y, ratio=2/3)

    # ensemble de valeur a tester pour l'hyperparametre kmax
    kmax = np.arange(25)

    # appliquer l'algorithme
    results = np.zeros((len(kmax), 3))
    for i, k in enumerate(kmax):
        omp = LinearRegressionOmp(n_iter=k)
        omp.fit(X_train, y_train)

        # error on learning set
        y_hat_train = omp.predict(X_train)
        lse_train = np.linalg.norm(y_hat_train - y_train, ord=2)

        # error on validation set
        y_hat_valid1 = omp.predict(X_valid1)
        lse_valid1 = np.linalg.norm(y_hat_valid1 - y_valid1, ord=2)

        # save results
        results[i, :] = np.array([k, lse_train, lse_valid1])

    # best value of hyperparameter lambda
    best_kmax = int(results[np.argmin(results[:, 2]), 0])
    
    return LinearRegressionOmp(n_iter=best_kmax)
