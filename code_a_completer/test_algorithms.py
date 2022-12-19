#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Valentin Emiya, AMU & CNRS LIS
"""

import unittest
import numpy as np

from algorithms import normalize_dictionary, mp, omp, ridge_regression


class TestNormalizeDictionary(unittest.TestCase):
    def test_normalize_dictionary(self):
        n_rows, n_cols = 123, 234
        X = np.random.randn(n_rows, n_cols)
        X_normalized, normalization_coefs = normalize_dictionary(X)
        # Check dimensions and shapes
        np.testing.assert_array_equal(normalization_coefs.ndim, 1)
        np.testing.assert_array_equal(normalization_coefs.shape, n_cols)
        np.testing.assert_array_equal(X_normalized.ndim, 2)
        np.testing.assert_array_equal(X_normalized.shape, (n_rows, n_cols))
        # Check input/output invariant
        np.testing.assert_array_almost_equal(
                X, X_normalized * normalization_coefs)
        # Check norm of the columns in the output matrix
        np.testing.assert_array_almost_equal(
                np.linalg.norm(X_normalized, axis=0), 1)


class TestRidge(unittest.TestCase):
    def test_with_identity_dictionary(self):
        """ Test with particular case: X=identity matrix """
        # Choose dimensions
        n_samples = 20
        n_features = n_samples
        k0 = 11

        # Build dictionary
        X = np.eye(n_samples)

        # Build solution
        w_ref = np.zeros(n_features)
        w_ref[np.random.permutation(n_features)[:k0]] = np.random.randn(k0)

        # Build observation
        y = X @ w_ref

        for lambda_ridge in [0, 0.1, 2]:
            # Estimate solution
            w_est = ridge_regression(X=X, y=y, lambda_ridge=lambda_ridge)

            # check shapes
            np.testing.assert_array_equal(w_est.shape, w_ref.shape)
            # check w_est (X is identity)
            np.testing.assert_array_almost_equal(w_est,
                                                 1 / (1 + lambda_ridge) * y)

    def test_random_noisy_case(self):
        # Choose dimensions
        n_samples = 17
        n_features = 73
        k0 = 12
        lambda_ridge = 0.1

        # Build dictionary
        X = np.random.randn(n_samples, n_features)
        X /= np.linalg.norm(X, axis=0)

        # Build solution
        w_ref = np.zeros(n_features)
        w_ref[np.random.permutation(n_features)[:k0]] = np.random.randn(k0)

        # Build observation
        y = X @ w_ref + 1e-3 * np.random.randn(n_samples)

        # Estimate solution
        w_est = ridge_regression(X=X, y=y, lambda_ridge=lambda_ridge)

        # Check shapes
        np.testing.assert_array_equal(w_est.shape, w_ref.shape)

        # Check objective value is minimum (lower or equal to value at some
        # test points)
        def obj_fun(w):
            return 0.5 * np.linalg.norm(X @ w - y)**2 \
                   + lambda_ridge / 2 * np.linalg.norm(w)**2
        obj_est = obj_fun(w_est)
        self.assertLessEqual(obj_est, obj_fun(w_ref))
        w = w_est + np.random.randn(*w_est.shape) * 10**(-9)
        self.assertLessEqual(obj_est, obj_fun(w),
                             msg="This test may fail sometimes")


if __name__ == '__main__':
    unittest.main()
