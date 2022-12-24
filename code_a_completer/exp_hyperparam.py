"""bbllalalala"""


import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_data, randomize_data, split_data
from linear_regression import LinearRegressionRidge, LinearRegressionMp

# load data
X_labeled, y_labeled, X_unlabeled = load_data("data/YearPredictionMSD_100.npz")

N = 500
X_labeled, y_labeled, X_unlabeled = X_labeled[:500, :N], y_labeled[:N], X_unlabeled[:N]

X_train, y_train, X_valid, y_valid = split_data(X_labeled, y_labeled, ratio=2/3)


# Ridge Regression

# ensemble de valeur a tester pour l'hyperparametre lambda
lambdas = np.logspace(0.0001, 1, num=50)/10 

# appliquer l'algorithme
results_ridge = np.zeros((len(lambdas),3))
for i, l in enumerate(lambdas):
    ridge_reg = LinearRegressionRidge(lambda_ridge=l)
    ridge_reg.fit(X_train, y_train)

    # error on learning set
    y_hat_train_ridge = ridge_reg.predict(X_train)
    lse_train_ridge = np.mean(np.square(y_hat_train_ridge - y_train))

    # error on validation set
    y_hat_valid_ridge = ridge_reg.predict(X_valid)
    lse_valid_ridge = np.mean(np.square(y_hat_valid_ridge - y_valid))

    results_ridge[i,:] = np.array([l, lse_train_ridge, lse_valid_ridge])

fig = plt.figure()
ax = plt.axes()
ax.plot(results_ridge[:, 0], results_ridge[:, 1], label="training set", linestyle="-", color="green")
ax.plot(results_ridge[:, 0], results_ridge[:, 2], label="validation set", linestyle=":", color="blue")
ax.scatter(results_ridge[np.argmin(results_ridge[:, 2]), 0], np.min(results_ridge[:, 2]),
           color="red", label=f"best lambda = {round(results_ridge[np.argmin(results_ridge[:, 2]), 0],3)}")
plt.xlabel("Ridge regularisation coefficent (lambda)")
plt.ylabel("Least Square Error")
plt.legend(loc="upper right")
plt.title("Ridge Regression: \n Least square error value for different values \n" +
          "of ridge regularization coefficents")
plt.savefig("figures/hyperparam_ridge.png")


# Matching Poursuit

# ensemble de valeur a tester pour l'hyperparametre kmax
kmax = np.arange(200)

# appliquer l'algorithme
results_mp = np.zeros((len(kmax),3))
for i, k in enumerate(kmax):
    mp = LinearRegressionMp(n_iter=k)
    mp.fit(X_train, y_train)

    # error on learning set
    y_hat_train_mp = mp.predict(X_train)
    lse_train_mp = np.mean(np.square(y_hat_train_mp - y_train))

    # error on validation set
    y_hat_valid_mp = mp.predict(X_valid)
    lse_valid_mp = np.mean(np.square(y_hat_valid_mp - y_valid))

    results_mp[i,:] = np.array([k, lse_train_mp, lse_valid_mp])

fig = plt.figure()
ax = plt.axes()
ax.plot(results_mp[:, 0], results_mp[:, 1], label="training set", linestyle="-", color="green")
ax.plot(results_mp[:, 0], results_mp[:, 2], label="validation set", linestyle=":", color="blue")
ax.scatter(results_mp[np.argmin(results_mp[:, 2]), 0], np.min(results_mp[:, 2]),
           color="red", label=f"best kmax = {results_mp[np.argmin(results_mp[:, 2]), 0]}")
plt.xlabel("Matching Poursuit hyperparameter (kmax)")
plt.ylabel("Least Square Error")
plt.legend(loc="upper right")
plt.title("Matching Poursuit: \n Least square error value for different values \n" +
          "of the hyperparameter kmax")
plt.savefig("figures/hyperparam_mp.png")


# 9. Estimation des hyperparametres par validation croisee

def learn_all_with_RIDGE(X,y):
    """_summary_

    Args:
        X (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    X_train, y_train, X_valid1, y_valid1 = split_data(X, y, ratio=2/3)

    # ensemble de valeur a tester pour l'hyperparametre lambda
    lambdas = np.logspace(0.0001, 1, num=50)/10 

    # appliquer l'algorithme
    results = np.zeros((len(lambdas),3))
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

# test to delete
ridge = LinearRegressionRidge(lambda_ridge=l)
ridge.fit(X_labeled, y_labeled)
y_hat = ridge.predict(X_valid)
np.mean(np.square(y_hat - y_valid))

ridge = learn_all_with_RIDGE(X_train, y_train)
ridge.fit(X_train, y_train)
y_hat = ridge.predict(X_valid)
print(np.mean(np.square(y_hat - y_valid)))




def learn_all_with_MP(X,y):
    """_summary_

    Args:
        X (_type_): _description_
        y (_type_): _description_
    """

    X_train, y_train, X_valid1, y_valid1 = split_data(X, y, ratio=2/3)

    # ensemble de valeur a tester pour l'hyperparametre kmax
    kmax = np.arange(200)

    # appliquer l'algorithme
    results = np.zeros((len(kmax),3))
    for i, k in enumerate(kmax):
        mp = LinearRegressionMp(n_iter=k)
        mp.fit(X_train, y_train)

        # error on learning set
        y_hat_train = mp.predict(X_train)
        lse_train = np.mean(np.square(y_hat_train - y_train))

        # error on validation set
        y_hat_valid1 = mp.predict(X_valid1)
        lse_valid1 = np.mean(np.square(y_hat_valid1 - y_valid1))

        # save results
        results[i,:] = np.array([l, lse_train, lse_valid1])

    # best value of hyperparameter lambda
    best_kmax = int(results[np.argmin(results[:, 2]), 0])
    
    return LinearRegressionMp(n_iter=best_kmax)

# test to delete
mp = learn_all_with_MP(X_train, y_train)
mp.fit(X_train, y_train)
y_hat = mp.predict(X_valid)
print(np.mean(np.square(y_hat - y_valid)))