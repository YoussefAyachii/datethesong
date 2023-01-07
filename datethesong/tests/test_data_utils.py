#!/usr/bin/env python3.9

"""Testing data_utils.py functions"""

import unittest
import numpy as np

from datethesong.algorithms.data_utils import(
    load_data, randomize_data, split_data)


class TestLoadData(unittest.TestCase):
    """
    Nom du fichier défini comme un attribut accessible dans toutes les
    méthodes.
    """
    filename = 'datethesong/data/YearPredictionMSD_100.npz'

    def test_data_shape(self):
        """test the shape of each matrix"""

        # Valeurs de référence
        n_lab_ex_expected = 4578
        data_dim_expected = 90
        n_unlab_ex_expected = 2289

        Xl, yl, Xu = load_data(self.filename)

        # Test des types
        self.assertIsInstance(Xl, np.ndarray)
        self.assertIsInstance(yl, np.ndarray)
        self.assertIsInstance(Xu, np.ndarray)

        # Test du nombre de dimensions
        self.assertEqual(Xl.ndim, 2)
        self.assertEqual(yl.ndim, 1)
        self.assertEqual(Xu.ndim, 2)

        # Test des formats
        np.testing.assert_array_equal(Xl.shape,
                                      (n_lab_ex_expected, data_dim_expected))
        np.testing.assert_array_equal(yl.shape,
                                      n_lab_ex_expected)
        np.testing.assert_array_equal(Xu.shape,
                                      (n_unlab_ex_expected, data_dim_expected))

    def test_first_last_elements_y(self):
        """test the equality between the first and last column
        of the y_labelised vector"""
        y_first5r = np.array([1985, 2005, 1998, 1973, 1957])
        y_last5r = np.array([2006, 1960, 1976, 1988, 2003])

        Xl, yl, Xu = load_data(self.filename)

        np.testing.assert_array_equal(
            x=yl[:5], y=y_first5r,
            err_msg="Wrong values in yl(first 5 elements).")
        np.testing.assert_array_equal(
            x=yl[-5:], y=y_last5r,
            err_msg="Wrong values in yl(last 5 elements).")

    def test_first_last_elements_X(self):
        """test the equality between different elements in both
        X_labeled and X_unlabeled"""

        Xl_first_row_first2c = 0.898881, 1.305756
        Xl_last_row_first2c = 1.393745, 0.1967
        Xl_first_row_lastc = -0.3353498396562418
        Xl_last_row_lastc = -0.025671111438439326

        Xu_first_row_first2c = 1.567927, 0.752978
        Xu_last_row_first2c = -0.256243, 0.481008
        Xu_first_row_lastc = 0.5988862660840178
        Xu_last_row_lastc = 0.3787646988802795

        Xl, yl, Xu = load_data(self.filename)

        np.testing.assert_array_almost_equal(
            x=Xl[0, :2], y=Xl_first_row_first2c, decimal=6,
            err_msg="Wrong values in Xl (first row, first 2 columns).")
        np.testing.assert_array_almost_equal(
            x=Xl[-1, :2], y=Xl_last_row_first2c, decimal=6,
            err_msg="Wrong values in Xl (last row, first 2 columns).")
        np.testing.assert_array_almost_equal(
            x=Xl[0, -1], y=Xl_first_row_lastc,
            err_msg="Wrong values in Xl (first row, last column).")
        np.testing.assert_array_almost_equal(
            x=Xl[-1, -1], y=Xl_last_row_lastc,
            err_msg="Wrong values in Xl (last row, last column).")

        np.testing.assert_array_almost_equal(
            x=Xu[0, :2], y=Xu_first_row_first2c, decimal=6,
            err_msg="Wrong values in Xu (first row, first 2 columns).")
        np.testing.assert_array_almost_equal(
            x=Xu[-1, :2], y=Xu_last_row_first2c, decimal=6,
            err_msg="Wrong values in Xu (last row, first 2 columns).")
        np.testing.assert_array_almost_equal(
            x=Xu[0, -1], y=Xu_first_row_lastc,
            err_msg="Wrong values in Xu (first row, last column)")
        np.testing.assert_array_almost_equal(
            x=Xu[-1, -1], y=Xu_last_row_lastc,
            err_msg="Wrong values in Xu (last row, last column)")


class TestRandomizeData(unittest.TestCase):
    def test_randomize_data(self):
        n_examples = 6
        data_dim = 7
        X = np.random.permutation(n_examples*data_dim)
        X = X.reshape(n_examples, data_dim)
        y = np.random.permutation(n_examples) * 10

        Xr, yr = randomize_data(X, y)

        # Test dimensions
        np.testing.assert_array_equal(
                Xr.shape, X.shape, "Xr and X do not have the same shape.")
        np.testing.assert_array_equal(
                yr.shape, y.shape, "yr and y do not have the same shape.")

        # Test contents
        for i in range(n_examples):
            # Find indices of y[i] in yr
            I = np.nonzero(yr == y[i])[0]
            # Test right number of occurences
            np.testing.assert_equal(
                I.size, 1,
                'Element {} found {} time(s) should occur exactly once in yr.'
                .format(y[i], I.size))
            # Test X and Xr contents for example i
            np.testing.assert_array_equal(
                X[i, :], Xr[I[0], :],
                'Row {} in X and row {} in Xr should be equal.'.format(i, I))

        # Test exception
        y_bad = y[:-1]
        with self.assertRaises(
                ValueError,
                msg="X with {} rows and y with {} elements"
                .format(X.shape[0], y_bad.size)):
            Xr, yr = randomize_data(X, y_bad)
        X_bad = X.T
        with self.assertRaises(
                ValueError,
                msg="X with {} rows and y with {} elements"
                .format(X_bad.shape[0], y.size)):
            Xr, yr = randomize_data(X_bad, y)


def _str_X_y(X, y):
    """ Auxiliary function """
    s = ''
    for i in range(X.shape[0]):
        s += 'y_{}={}, x_{}={}\n'.format(i, y[i], i, X[i, :])
    return s


class TestSplitData(unittest.TestCase):
    def test_split_data(self):
        # Choose dimensions
        n = 10
        d = 5
        # Generate data
        X = np.arange(n*d).reshape((n, d)) * 10
        y = -np.arange(n)*2 - 5
        s = 'Original data\n' + _str_X_y(X, y)  # string for debug

        # Split data
        n1 = 7
        n2 = n - n1
        ratio = n1 / n
        X1, y1, X2, y2 = split_data(X, y, ratio)
        s += '(X1, y1):\n' + _str_X_y(X1, y1)
        s += '(X2, y2):\n' + _str_X_y(X2, y2)

        # Test types
        self.assertIsInstance(X1, np.ndarray, msg='X1 is not an array\n'+s)
        self.assertIsInstance(X2, np.ndarray, msg='X2 is not an array\n'+s)
        self.assertIsInstance(y1, np.ndarray, msg='y1 is not an array\n'+s)
        self.assertIsInstance(y2, np.ndarray, msg='y2 is not an array\n'+s)

        # Test number of dimensions
        self.assertEqual(
                X1.ndim, 2,
                msg='X1 should be a matrix, found shape {}.\n{}'
                .format(X1.shape, s))
        self.assertEqual(
                X2.ndim, 2,
                msg='X2 should be a matrix, found shape {}.\n{}'
                .format(X2.shape, s))
        self.assertEqual(
                y1.ndim, 1,
                msg='y1 should be a vector, found shape {}.\n{}'
                .format(y1.shape, s))
        self.assertEqual(
                y2.ndim, 1,
                msg='y2 should be a vector, found shape {}.\n{}'
                .format(y2.shape, s))

        # Test shapes
        np.testing.assert_array_equal(
                X1.shape, [n1, d],
                err_msg='X1 should have shape {}, found {}.\n{}'
                .format([n1, d], X1.shape, s))
        np.testing.assert_array_equal(
                X2.shape, [n2, d],
                err_msg='X2 should have shape {}, found {}.\n{}'
                .format([n2, d], X2.shape, s))
        np.testing.assert_array_equal(
                y1.shape, [n1],
                err_msg='y1 should have shape {}, found {}.\n{}'
                .format([n1], y1.shape, s))
        np.testing.assert_array_equal(
                y2.shape, [n2],
                err_msg='y2 should have shape {}, found {}.\n{}'
                .format([n2], y2.shape, s))

        # Test union
        Xu = np.concatenate((X1, X2), axis=0)
        yu = np.concatenate((y1, y2), axis=0)
        yX = np.concatenate((y[:, None], X), axis=1)
        yXu = np.concatenate((yu[:, None], Xu), axis=1)
        for i in range(n):
            # find yX[i, :] in yXu
            row_tests = np.all(yX[i, :][None, :] == yXu, axis=1)
            self.assertTrue(
                    np.sum(row_tests) >= 1,
                    msg='y={} x={} (i={}) not found in partition.\n{}'
                    .format(yX[i, 0], yX[i, 1:], i, s))
            # find yXu[i, :] in yX
            row_tests = np.all(yXu[i, :][None, :] == yX, axis=1)
            self.assertTrue(
                    np.sum(row_tests) >= 1,
                    msg='y={} x={} (i={}) not found in original data.\n{}'
                    .format(yXu[i, 0], yXu[i, 1:], i, s))


if __name__ == '__main__':
    unittest.main()
