import numpy as np
from data_utils import load_data

"""Load training data and print dimensions as well as a few coefficients
in the first and last places and at random locations."""

X_labeled, y_labeled, X_unlabeled = load_data("data/YearPredictionMSD_100.npz")

print(f"X_labeled is a {type(X_labeled)} of dimensions {X_labeled.ndim}, shape {X_labeled.shape} and enclosing data of type {X_labeled.dtype}")
print(f"y_labeled is a {type(y_labeled)}  of dimensions {y_labeled.ndim}, shape {y_labeled.shape}  and enclosing data of type {y_labeled.dtype}")
print(f"X_unlabeled is a {type(X_unlabeled)} of dimensions {X_unlabeled.ndim}, shape {X_unlabeled.shape}  and enclosing data of type {X_unlabeled.dtype}")

print("\n first 5 values of y_labeled: \n", y_labeled[:5])
print("\n last 5 values of y_labeled: \n", y_labeled[-5:])

# 2 premiers coefficients de la première ligne et de la dernière ligne de chaque matrice,
print(f"\n 2 premiers coefficients de la première ligne et de la dernière ligne de X_labeled: \n {X_labeled[[0,-1],0:2]}")
print(f"\n 2 premiers coefficients de la première ligne et de la dernière ligne de X_unlabeled: \n {X_unlabeled[[0,-1],0:2]}")

# le dernier coefficient de la première et la dernière ligne
print(f"\n le dernier coefficient de la première et la dernière ligne de X_labeled: \n {X_labeled[[0,-1],-1]}")
print(f"\n le dernier coefficient de la première et la dernière ligne de X_unlabeled: \n {X_unlabeled[[0,-1],-1]}")
