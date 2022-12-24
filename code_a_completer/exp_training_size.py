"""Mesurer les performances de l'estimateur constant majoritaire
for datasets of different size"""

import numpy as np
from time import time
from data_utils import load_data, split_data

# Part 1: Choisisser une strategie d'estimation constante: regle majoritaire

# load data
X_labeled, y_labeled, X_unlabeled = load_data("data/YearPredictionMSD_100.npz")

# Building randomly: S_train and S_valid
S_train_X, S_train_y, S_valid_X, S_valid_y = split_data(X_labeled, y_labeled, 2/3)

# Ensemble N
N = np.array([2**i for i in np.arange(6,11+1)])

# Initializing train_error and train_valid: to stock prediction errors for each N
train_error, valid_error, learning_time = np.zeros(len(N)), np.zeros(len(N)), np.zeros(len(N))

# Iteration
for i,n in enumerate(N):
    s_train_X, s_train_y = S_train_X[0:n, :], S_train_y[0:n]
    s_valid_X, s_valid_y = S_valid_X[0:n, :], S_valid_y[0:n]
    
    # Linear Regression Majority 
    # 1. appliquer algo d'apprentissage a s_train_X
    majority_algo = LinearRegressionMajority()
    t0 = time()
    majority_algo.fit(s_train_X, s_train_y)
    t1 = time()
    
    # y_hat: predicted labels on s_valid_X
    s_valid_yhat = majority_algo.predict(s_valid_X)
    
    # prediction error on s_valid
    quadratic_error_valid = np.sum(np.square(s_valid_yhat - s_valid_y))
    valid_error[i] = quadratic_error_valid
    
    # prediction error on s_train
    s_train_yhat = majority_algo.predict(s_train_X)
    quadratic_error_train = np.sum(np.square(s_train_yhat - s_train_y))
    train_error[i] = quadratic_error_train
    learning_time[i] = t1 - t0

np.savez("exp_training_size_results.npz",
         N=N,
         train_error=train_error,
         valid_error=valid_error
         )
# to load train_error array for ex:
# np.load("exp_training_size_results_maj.npz")["train_error"]

# Part 2: Estimer les performances des 3 strategies constantes

def get_valid_and_train_errors(cte_lr_method,
                               s_train_X,
                               s_train_y,
                               s_valid_X,
                               s_valid_y):
    """apply a linear regression method
    
    Parameters:
        lr_method (_type_): _description_
        s_train_X (_type_): _description_
        s_train_y (_type_): _description_
        s_valid_X (_type_): _description_
        s_valid_y (_type_): _description_
    """
    # appliquer algo d'apprentissage a s_train_X
    t0 = time()
    cte_lr_method.fit(s_train_X, s_train_y)
    t1 = time()
    # y_hat: predicted labels on s_valid_X
    s_valid_yhat = cte_lr_method.predict(s_valid_X)
    
    # y_hat: predicted labels on s_train_X
    s_train_yhat = cte_lr_method.predict(s_train_X)

    # prediction error on s_valid
    MSE_valid = np.sum(np.square(s_valid_yhat - s_valid_y))
    MSE_train = np.sum(np.square(s_train_yhat - s_train_y))
    computation_time = t1 - t0
    
    return (MSE_valid, MSE_train, computation_time)


for i,n in enumerate(N):
    s_train_X, s_train_y = S_train_X[0:n, :], S_train_y[0:n]
    s_valid_X, s_valid_y = S_valid_X[0:n, :], S_valid_y[0:n]
    
    MSE_valid_maj, MSE_train_maj, learning_time_majority = get_valid_and_train_errors(
        cte_lr_method=LinearRegressionMajority(), s_train_X=s_train_X, s_train_y=s_train_y, s_valid_X=s_valid_X, s_valid_y=s_valid_y)
    MSE_valid_median, MSE_train_median, learning_time_median = get_valid_and_train_errors(
        cte_lr_method=LinearRegressionMedian(), s_train_X=s_train_X, s_train_y=s_train_y, s_valid_X=s_valid_X, s_valid_y=s_valid_y)
    MSE_valid_mean, MSE_train_mean, learning_time_mean = get_valid_and_train_errors(
        cte_lr_method=LinearRegressionMean(), s_train_X=s_train_X, s_train_y=s_train_y, s_valid_X=s_valid_X, s_valid_y=s_valid_y)
    
    MSE_valid_temp = np.array([MSE_valid_maj, MSE_valid_median, MSE_valid_mean])
    MSE_train_temp = np.array([MSE_train_maj, MSE_train_median, MSE_train_mean])
    learning_time_temp = np.array([learning_time_majority, learning_time_median, learning_time_mean])
    
    if i == 0:
        # create np.arrays
        valid_error, train_error, learning_time = MSE_valid_temp, MSE_train_temp, learning_time_temp
    else:
        # add by row
        valid_error = np.vstack((valid_error,  MSE_valid_temp))
        train_error = np.vstack((train_error, MSE_train_temp))
        learning_time = np.vstack((learning_time, learning_time_temp))
        
print("\n valid error: \n ", valid_error)
print("\n train error: \n ", train_error)
print("\n N vector: \n", N)
print("\n learning_time: \n", learning_time)

# update the npz file
np.savez("exp_training_size_results.npz",
         N=N,
         train_error=train_error,
         valid_error=valid_error,
         learning_time=learning_time
         )

# Part 3: Least Square
ls_train_error, ls_valid_error, ls_learning_time = np.zeros(len(N)), np.zeros(len(N)), np.zeros(len(N))

for i,n in enumerate(N):
    s_train_X, s_train_y = S_train_X[0:n, :], S_train_y[0:n]
    s_valid_X, s_valid_y = S_valid_X[0:n, :], S_valid_y[0:n]
    
    # appliquer algo d'apprentissage LR LeastSquare
    ls_algo = LinearRegressionLeastSquares()
    t0 = time()
    ls_algo.fit(s_train_X, s_train_y)
    t1 = time()
    
    # y_hat: predicted labels on s_valid_X
    s_valid_yhat = ls_algo.predict(s_valid_X)
    
    # prediction error on s_valid
    MSE_valid = np.sum(np.square(s_valid_yhat - s_valid_y))
    ls_valid_error[i] = MSE_valid
    
    # prediction error on s_train
    s_train_yhat = ls_algo.predict(s_train_X)
    MSE_train = np.sum(np.square(s_train_yhat - s_train_y))
    ls_train_error[i] = MSE_train
    
    # computation time
    ls_learning_time[i] = t1 - t0

# add Least Square perfomences to valid_error and train_error matrices
valid_error = np.column_stack((valid_error, ls_valid_error))
train_error = np.column_stack((train_error, ls_train_error))
learning_time = np.column_stack((learning_time, ls_learning_time))

print("\n valid error: \n ", valid_error)
print("\n train error: \n ", train_error)
print("\n N vector: \n", N)
print("\n learning_time: \n", learning_time)


# update the npz file
np.savez("exp_training_size_results.npz",
         N=N,
         train_error=train_error,
         valid_error=valid_error,
         learning_time=learning_time)

