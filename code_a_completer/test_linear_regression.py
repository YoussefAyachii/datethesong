#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Valentin Emiya, AMU & CNRS LIS
"""

import unittest
import numpy as np

from linear_regression import LinearRegression, LinearRegressionLeastSquares, \
    LinearRegressionMean, LinearRegressionMedian, LinearRegressionMajority, \
    LinearRegressionRidge, LinearRegressionMp, LinearRegressionOmp


class TestLinearRegression(unittest.TestCase):
    def test_predict(self):
        # Generate data
        n = 3
        d = 4
        w = np.array([1, -3, 100, 1.6])
        b = -13
        X = np.arange(n * d).reshape((n, d))
        y_expected = np.array([188.8, 587.2, 985.6])

        linear_regression = LinearRegression()
        linear_regression.w = w
        linear_regression.b = b
        y = linear_regression.predict(X)

        # Test output values
        np.testing.assert_array_almost_equal(y, y_expected)

    def test_predict_errors(self):
        n = 3
        d = 4
        X = np.arange(n * d).reshape((n, d))
        y_expected = np.array([188.8, 587.2, 985.6])

        linear_regression = LinearRegression()
        # Case w and b not set (None)
        with self.assertRaises(AttributeError):
            _ = linear_regression.predict(X)

        linear_regression.w = np.array([1, -3, 100, 1.6])
        linear_regression.b = -13

        # Case with bad argument X
        with self.assertRaises(AssertionError):
            _ = linear_regression.predict(y_expected)
        with self.assertRaises(AssertionError):
            _ = linear_regression.predict(list(X))
        with self.assertRaises(AssertionError):
            _ = linear_regression.predict(X[0, :])


class TestLinearRegressionLeastSquares(unittest.TestCase):
    def test_fit(self):
        # Generate data
        n = 10
        d = 5
        w = np.random.randn(d)
        b = np.random.randn()
        X = np.random.randn(n, d)
        y = X @ w + b

        linear_reg = LinearRegressionLeastSquares()
        linear_reg.fit(X, y)

        # Test output values
        np.testing.assert_array_almost_equal(
            linear_reg.b, b,
            err_msg='Constant coef is not correctly estimated')
        np.testing.assert_array_almost_equal(
            linear_reg.w, w,
            err_msg='Weight vector is not correctly estimated')


class TestLinearRegressionMean(unittest.TestCase):
    def test_fit(self):
        # Generate data
        n = 10
        d = 5
        X = np.random.randn(n, d)
        y = np.array([5, 4, 6, -20, 4, 6, 6, 2, 1, 1])
        b_expected = 1.5

        linear_reg = LinearRegressionMean()
        linear_reg.fit(X, y)

        # Test output values
        np.testing.assert_array_almost_equal(
            linear_reg.b, b_expected,
            err_msg='Constant coef is not correctly estimated')
        np.testing.assert_array_equal(
            linear_reg.w, 0,
            err_msg='Weight vector is not correctly estimated')


class TestLinearRegressionMedian(unittest.TestCase):
    def test_fit(self):
        # Generate data
        n = 10
        d = 5
        X = np.random.randn(n, d)
        y = np.array([5, 4, 6, -20, 4, 6, 6, 2, 1, 1])
        b_expected = 4

        linear_reg = LinearRegressionMedian()
        linear_reg.fit(X, y)

        # Test output values
        np.testing.assert_array_almost_equal(
            linear_reg.b, b_expected,
            err_msg='Constant coef is not correctly estimated')
        np.testing.assert_array_equal(
            linear_reg.w, 0,
            err_msg='Weight vector is not correctly estimated')


class TestLinearRegressionMajority(unittest.TestCase):
    def test_fit(self):
        # Generate data
        n = 10
        d = 5
        X = np.random.randn(n, d)
        y = np.array([5, 4, 6, -20, 4, 6, 6, 2, 1, 1])
        b_expected = 6

        linear_reg = LinearRegressionMajority()
        linear_reg.fit(X, y)

        # Test output values
        np.testing.assert_array_almost_equal(
            linear_reg.b, b_expected,
            err_msg='Constant coef is not correctly estimated')
        np.testing.assert_array_equal(
            linear_reg.w, 0,
            err_msg='Weight vector is not correctly estimated')

"""
class TestLinearRegressionRidge(unittest.TestCase):
    def test_fit_identity(self):
        # Choose dimensions
        n_samples = 20
        n_features = n_samples
        support = [0, 2, 4, 5, 10]

        # Build dictionary
        X = np.eye(n_samples)

        # Build solution
        w_ref = np.zeros(n_features)
        w_ref[support] = np.random.randn(len(support))
        b_ref = 100

        # Build observation
        y = X @ w_ref + b_ref

        for lambda_ridge in [0, 0.1, 1, 2]:
            b_expected = np.mean(y)
            w_expected = 1 / (1 + lambda_ridge) * (y - b_expected)

            # Estimate solution
            ridge_reg = LinearRegressionRidge(lambda_ridge=lambda_ridge)
            ridge_reg.fit(X=X, y=y)
            w_est = ridge_reg.w
            b_est = ridge_reg.b

            self.assertTrue(b_est is not None)
            # check shapes
            np.testing.assert_array_equal(w_est.shape, w_ref.shape)
            # check w
            np.testing.assert_array_almost_equal(w_est, w_expected)
            # check b
            np.testing.assert_almost_equal(b_est, b_expected)

    def test_fit_objective_value(self):
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
        y -= np.mean(y)  # center y so that b = 0

        # Estimate solution
        ridge_reg = LinearRegressionRidge(lambda_ridge=lambda_ridge)
        ridge_reg.fit(X=X, y=y)
        w_est = ridge_reg.w
        b_est = ridge_reg.b

        # Check shapes
        np.testing.assert_array_equal(w_est.shape, w_ref.shape)

        # Check b
        self.assertAlmostEqual(b_est, 0)

        # Check objective value is minimum (lower or equal to value at some
        # test points)

        def obj_fun(w, b):
            return 0.5 * np.linalg.norm(X @ w + b - y) ** 2 \
                   + lambda_ridge / 2 * np.linalg.norm(w) ** 2

        obj_est = obj_fun(w_est, b_est)
        self.assertLessEqual(obj_est, obj_fun(w_ref, 0))
        for i in range(3):
            w = w_est + np.random.randn(*w_est.shape) * 10 ** (-i)
            b = b_est + np.random.randn() * 10 ** (-i)
            self.assertLessEqual(
                obj_est, obj_fun(w, b),
                msg="This test may fail sometimes")


class TestLinearRegressionMp(unittest.TestCase):
    def test_fit_identity(self):
        # Choose dimensions
        n_samples = 20
        n_features = n_samples
        support = [0, 2, 4, 5, 10]
        n_iter = n_features

        # Build dictionary
        X = np.eye(n_samples)
        X -= np.mean(X, axis=0)  # center each column
        X *= np.random.rand(n_features) + 10  # force denormalization

        # Build solution
        w_ref = np.zeros(n_features)
        w_ref[support] = np.random.randn(len(support))
        b_ref = 100

        # Build observation
        y = X @ w_ref + b_ref

        # Estimate solution
        mp_reg = LinearRegressionMp(n_iter=n_iter)
        mp_reg.fit(X=X, y=y)
        w_est = mp_reg.w
        b_est = mp_reg.b

        # check shapes
        np.testing.assert_array_equal(w_est.shape, w_ref.shape)
        # w_est should match w_ref approximately
        np.testing.assert_array_almost_equal(w_est, w_ref, decimal=2)
        # check b
        np.testing.assert_almost_equal(b_est, b_ref)

    def test_fit_dct_recovery(self):
        # Choose dimensions
        n_samples = 13
        n_features = 15
        support = [1, 2, 8]
        n_iter = len(support)

        # Build dictionary (DCT-IV)
        X = np.cos(np.pi
                   * (np.arange(n_samples)[:, None] + 0.5)
                   * (np.arange(n_features)[None, :] + 0.5)
                   / n_features)
        X -= np.mean(X, axis=0)  # center each column
        X /= np.linalg.norm(X, axis=0)  # normalize
        X *= np.random.rand(n_features)[None, :] + 10  # force denormalization

        # Build solution
        w_ref = np.zeros(n_features)
        w_ref[support] = 1
        b_ref = -1

        # Build observation
        y = X @ w_ref + b_ref

        # Estimate solution
        mp_reg = LinearRegressionMp(n_iter=n_iter)
        mp_reg.fit(X=X, y=y)
        w_est = mp_reg.w
        b_est = mp_reg.b

        # Check shapes
        np.testing.assert_array_equal(w_est.shape, w_ref.shape)
        # Check support
        np.testing.assert_array_equal(np.nonzero(w_est), np.nonzero(w_ref))
        # check b
        np.testing.assert_almost_equal(b_est, b_ref)

    def test_fit_noisy(self):
        # Choose dimensions
        n_samples = 17
        n_features = 73
        k0 = 12
        n_iter = n_samples

        # Build dictionary
        X = np.random.randn(n_samples, n_features)
        X -= np.mean(X, axis=0)  # center each column
        X *= np.random.rand(n_features)  # force denormalization

        # Build solution
        w_ref = np.zeros(n_features)
        w_ref[np.random.permutation(n_features)[:k0]] = np.random.randn(k0)
        b_ref = 5

        # Build observation
        y = X @ w_ref + b_ref + 1e-9 * np.random.randn(n_samples)

        # Estimate solution
        mp_reg = LinearRegressionMp(n_iter=n_iter)
        mp_reg.fit(X=X, y=y)
        w_est = mp_reg.w
        b_est = mp_reg.b

        # check shapes
        np.testing.assert_array_equal(w_est.shape, w_ref.shape)
        # check b
        np.testing.assert_almost_equal(b_est, b_ref)


class TestLinearRegressionOmp(unittest.TestCase):
    def test_fit_identity(self):
        # Choose dimensions
        n_samples = 20
        n_features = n_samples
        k0 = 5
        n_iter = n_features

        # Build dictionary
        X = np.eye(n_samples)
        X -= np.mean(X, axis=0)  # center each column
        X *= np.random.rand(n_features) + 10  # force denormalization

        # Build solution
        w_ref = np.zeros(n_features)
        w_ref[np.random.permutation(n_features)[:k0]] = np.random.randn(k0)
        b_ref = -10

        # Build observation
        y = X @ w_ref + b_ref

        # Estimate solution
        omp_reg = LinearRegressionOmp(n_iter=n_iter)
        omp_reg.fit(X=X, y=y)
        w_est = omp_reg.w
        b_est = omp_reg.b

        # check shapes
        np.testing.assert_array_equal(w_est.shape, w_ref.shape)
        # w_est should match w_ref since X is identity
        np.testing.assert_array_almost_equal(w_est, w_ref)
        # check b
        np.testing.assert_almost_equal(b_est, b_ref)

    def test_fit_dct_recovery(self):
        # Choose dimensions
        n_samples = 13
        n_features = 15
        support = [1, 2, 8]
        n_iter = len(support)

        # Build dictionary (DCT-IV)
        X = np.cos(np.pi
                   * (np.arange(n_samples)[:, None] + 0.5)
                   * (np.arange(n_features)[None, :] + 0.5)
                   / n_features)
        X -= np.mean(X, axis=0)  # center each column
        X /= np.linalg.norm(X, axis=0)  # normalize
        X *= np.random.rand(n_features)[None, :] + 10  # force denormalization

        # Build solution
        w_ref = np.zeros(n_features)
        w_ref[support] = 1
        b_ref = -20

        # Build observation
        y = X @ w_ref + b_ref

        # Estimate solution
        omp_reg = LinearRegressionOmp(n_iter=n_iter)
        omp_reg.fit(X=X, y=y)
        w_est = omp_reg.w
        b_est = omp_reg.b

        # check shapes
        np.testing.assert_array_equal(w_est.shape, w_ref.shape)
        # Check support
        np.testing.assert_array_equal(np.nonzero(w_est), np.nonzero(w_ref))
        # w_est should match w_ref due to orthoprojection
        np.testing.assert_array_almost_equal(w_est, w_ref)
        # check b
        np.testing.assert_almost_equal(b_est, b_ref)

    def test_fit_noisy(self):
        # Choose dimensions
        n_samples = 17
        n_features = 73
        k0 = 12
        n_iter = n_samples

        # Build dictionary
        X = np.random.randn(n_samples, n_features)
        X -= np.mean(X, axis=0)  # center each column
        X *= np.random.rand(n_features)  # force denormalization

        # Build solution
        w_ref = np.zeros(n_features)
        w_ref[np.random.permutation(n_features)[:k0]] = np.random.randn(k0)
        b_ref = 23

        # Build observation
        y = X @ w_ref + b_ref + 1e-9 * np.random.randn(n_samples)

        # Estimate solution
        omp_reg = LinearRegressionOmp(n_iter=n_iter)
        omp_reg.fit(X=X, y=y)
        w_est = omp_reg.w
        b_est = omp_reg.b

        # check shapes
        np.testing.assert_array_equal(w_est.shape, w_ref.shape)
        # check b
        np.testing.assert_almost_equal(b_est, b_ref)
"""

if __name__ == '__main__':
    unittest.main()
