#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Valentin Emiya, AMU & CNRS LIS
"""
import numpy as np


def normalize_dictionary(X):
    """
    Normalize matrix to have unit l2-norm columns

    Parameters
    ----------
    X : np.ndarray [n, d]
        Matrix to be normalized

    Returns
    -------
    X_normalized : np.ndarray [n, d]
        Normalized matrix
    norm_coefs : np.ndarray [d]
        Normalization coefficients (i.e., l2-norm of each column of ``X``)
    """
    # Check arguments
    assert isinstance(X, np.ndarray)
    assert X.ndim == 2

    # TODO À compléter
    # to vectorize: nympy multiplicatio not loop !!
    X_tild = np.zeros((X.shape))
    norm_coefs = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        a_m = X[:, i]
        a_m_l2 = np.linalg.norm(a_m)
        a_m_tild = a_m / a_m_l2
        
        X_tild[:, i] = a_m_tild
        norm_coefs[i] = a_m_l2
        # norme euclidienne des a_m de X_tild doit etre unitaire ???
    return X_tild, norm_coefs
    
    
        


def ridge_regression(X, y, lambda_ridge):
    """
    Ridge regression estimation

    Minimize $\left\| X w - y\right\|_2^2 + \lambda \left\|w\right\|_2^2$
    with respect to vector $w$, for $\lambda > 0$ given a matrix $X$ and a
    vector $y$.

    Note that no constant term is added.

    Parameters
    ----------
    X : np.ndarray [n, d]
        Data matrix composed of ``n`` training examples in dimension ``d``
    y : np.ndarray [n]
        Labels of the ``n`` training examples
    lambda_ridge : float
        Non-negative penalty coefficient

    Returns
    -------
    w : np.ndarray [d]
        Estimated weight vector
    """
    # Check arguments
    assert X.ndim == 2
    n_samples, n_features = X.shape
    assert y.ndim == 1
    assert y.size == n_samples

    # TODO À compléter
    Xt_X = X.T @ X
    lambda_id = lambda_ridge * np.eye(n_features)
    w = (Xt_X + lambda_id) @ X.T @ y
    return w

def mp(X, y, n_iter):
    """
    Matching pursuit algorithm

    Parameters
    ----------
    X : np.ndarray [n, d]
        Dictionary, or data matrix composed of ``n`` training examples in
        dimension ``d``. It should be normalized to have unit l2-norm
        columns before calling the algorithm.
    y : np.ndarray [n]
        Observation vector, or labels of the ``n`` training examples
    n_iter : int
        Number of iterations

    Returns
    -------
    w : np.ndarray [d]
        Estimated sparse vector
    error_norm : np.ndarray [n_iter+1]
        Vector composed of the norm of the residual error at the beginning
        of the algorithm and at the end of each iteration
    """
    # Check arguments
    assert X.ndim == 2
    n_samples, n_features = X.shape
    assert y.ndim == 1
    assert y.size == n_samples

    # TODO À compléter
    # initialization
    r = y
    w = np.zeros(n_features)
    for i,k in enumerate(__):
        a_m = X[:, i]
        c_m = a_m @ r
        i, j = np.unravel_index(c_m.argmax(), c_m.shape) # ??
        m_hat = c_m[i, j]
        # update w and r
        w[m_hat] = w[m_hat] + c_m
        r = r - (c_m * a_m)
        
    return w, error_norm # error_norm : vecteur de taille n_iter+1

def omp(X, y, n_iter):
    """
    Orthogonal matching pursuit algorithm

    Parameters
    ----------
    X : np.ndarray [n, d]
        Dictionary, or data matrix composed of ``n`` training examples in
        dimension ``d``. It should be normalized to have unit l2-norm
        columns before calling the algorithm.
    y : np.ndarray [n]
        Observation vector, or labels of the ``n`` training examples
    n_iter : int
        Number of iterations

    Returns
    -------
    w : np.ndarray [d]
        Estimated sparse vector
    error_norm : np.ndarray [n_iter+1]
        Vector composed of the norm of the residual error at the beginning
        of the algorithm and at the end of each iteration
    """
    # Check arguments
    assert X.ndim == 2
    n_samples, n_features = X.shape
    assert y.ndim == 1
    assert y.size == n_samples

    # TODO À compléter
