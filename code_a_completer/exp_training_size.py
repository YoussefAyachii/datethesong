"""measuring the performance of the estimateur constant majoritaire
for datasets of different size"""

import numpy as np
from data_utils import load_data, split_data
from linear_regression import LinearRegressionMajority

# load data
X_labeled, y_labeled, X_unlabeled = load_data("data/YearPredictionMSD_100.npz")

# Building randomly: S_train and S_valid
S_train_X, S_train_y, S_valid_X, S_valid_y = split_data(X_labeled, y_labeled, 2/3)

# Ensemble N
N = np.flip(np.array([2**i for i in range(5+1)]))

# Building train_error and train_valid
train_error, valid_error = np.zeros(len(N)), np.zeros(len(N))

# Iteration
for i,n in enumerate(N):
    # selectionner les N premiers example de S_train
    s_train_X, s_train_y = S_train_X[0:n], S_train_y[0:n]
    #print(s_train_X.shape, s_train_y.shape)
    
    # appliquer l'algo
    majority = LinearRegressionMajority()
    #print(type(s_train_X), type(s_train_y))
    w, b = majority.fit(s_train_X, s_train_y)
    #print( w.shape, type(w), b)
    
    
    # Estimation of S_valid labels ??????
    s_valid_X, s_valid_y = S_valid_X[0:n], S_valid_y[0:n]
    # print(type(s_valid_X), type(s_valid_y))
    pred = majority.predict(s_valid_X)
    #print(pred)
    

    """
    # Erreur quadratique moyenne sur Svalid et stocker-la ?????
    mse = mean(np.square(y - y_hat)) # ????
    valid_error[i] = mse #?? 
    """