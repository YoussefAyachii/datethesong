#!/usr/bin/env python3.9

"""Regularization methods for linear regression problems"""

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

    rows, cols = X.shape
    X_tild = np.zeros(X.shape)
    norm_coefs = np.zeros(cols)

    norm_coefs = np.sqrt(np.sum(np.square(X), axis=0))
    X_tild = np.divide(X, norm_coefs)

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

    Xt_X = X.T @ X
    lambda_id = lambda_ridge * np.eye(n_features)
    w = np.linalg.inv(Xt_X + lambda_id) @ X.T @ y
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

    # initialization
    r = y
    w = np.zeros(n_features)
    error_norm = np.zeros(n_iter + 1)
    
    # first residual error
    error_norm[0] = np.linalg.norm(y - X @ w, ord=2)

    for k in range(n_iter):
        C = X.T @ r
        # m_hat: index of max value in C
        m_hat = np.argmax(abs(C))
        
        # update w and r
        w[m_hat] = w[m_hat] + C[m_hat]
        r = r - C[m_hat] * X[:, m_hat]
        
        # save temp residual value 
        error_norm[k + 1] = np.linalg.norm(r, ord=2)

        # verify that the norm decreases
        assert error_norm[k + 1] <= error_norm[k]
        
        # verification: r orthogonal to a_m_hat
        np.testing.assert_array_almost_equal(r @ X[:, m_hat], 0, decimal=10)

    return w, error_norm

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

    r = y
    w = np.zeros(n_features)
    error_norm = np.zeros(n_iter + 1)
    Omega = np.array([])

    # first residual error
    error_norm[0] = np.linalg.norm(y - X @ w, ord=2)

    for k in range(n_iter):
        C = X.T @ r
        # m_hat: index of max value in C
        m_hat = np.argmax(abs(C))

        # update
        Omega = np.append(Omega, int(m_hat))
        w[Omega.astype(int)] = np.linalg.pinv(X[:, Omega.astype(int)]) @ y
        r = y - X @ w

        # save temp residual value
        error_norm[k + 1] = np.linalg.norm(r, ord=2)

        # verify in each step that the norm decreases
        # assert error_norm[k + 1] <= error_norm[k]

        # verify orthogonality at each step
        # np.testing.assert_array_almost_equal(
            # r @ X[:, Omega.astype(int)], 0, decimal=0)     

    return w, error_norm
