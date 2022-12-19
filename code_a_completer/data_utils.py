import numpy as np


def load_data(filename):
    """
    Load data examples from a npz file

    Parameters
    ----------
    filename : str
        Name of the npz file containing the data to be loaded

    Returns
    -------
    X_labeled : np.ndarray [n, d]
        Array of n feature vectors with size d
    y_labeled : np.ndarray [n]
        Vector of n labels related to the n feature vectors
    X_unlabeled :

    """
    npz_data = np.load(filename)
    dict_data = dict(npz_data)

    X_labeled = dict_data["X_labeled"]
    y_labeled = dict_data["y_labeled"]
    X_unlabeled = dict_data["X_unlabeled"]

    return X_labeled, y_labeled, X_unlabeled


def randomize_data(X, y):
    """
    Randomly shuffle the rows, i.e. observations, in the labeled set (X, y).

    Parameters
    ----------
    X : np.ndarray [n, d]
        Array of n feature vectors with size d
    y : np.ndarray [n]
        Vector of n labels related to the n feature vectors

    Returns
    -------
    Xr : np.ndarray [n, d]
        Shuffled version of X
    yr : np.ndarray [n]
        Shuffled version of y

    Raises
    ------
    ValueError
        If the number of rows in X differs from the number of elements in y.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError('Number of rows in X ({}) differs from the number of '
                         'elements in y'
                         .format(X.shape[0], y.shape[0]))
    else:
        n = X.shape[0]
        index_rand = np.random.permutation(n)
        Xr, yr = X[index_rand, :], y[index_rand]
        return Xr, yr

def split_data(X, y, ratio):
    """
    Split a set of n labeled examples into two subsets as a random partition.

    split_data(X, y, ratio) returns a tuple (X1, y1, X2, y2). The n input
    labeled examples (X,y) are randomly permuted and split as a partition
    {(X1, y1), (X2, y2)}. The respective size n1 and n2 is such that
    n1/n approximately equals the input argument `ratio` and n1+n2 = n.

    Parameters
    ----------
    X : np.ndarray [n, d]
        Array of n feature vectors with size d
    y : np.ndarray [n]
        Vector of n labels related to the n feature vectors
    ratio : float
        Ratio of data to be extracted into (X1, y1)

    Returns
    -------
    X1 : np.ndarray [n1, d]
        Array of n1 feature vectors
    y1 : np.ndarray [n1]
        Vector of n1 label
    X2 : np.ndarray [n2, d]
        Array of n2 feature vectors
    y2 : np.ndarray [n2]
        Vector of n2 labels selected

    """
    n = X.shape[0]
    n1 = int(ratio * n)
    n2 = n - n1

    Xr, yr = randomize_data(X, y)
    X1, X2 = Xr[:n1, ], Xr[n1:, ]
    y1, y2 = yr[:n1], yr[n1: ]
    
    return (X1, y1, X2, y2)