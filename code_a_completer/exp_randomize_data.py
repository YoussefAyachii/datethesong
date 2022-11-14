#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:03:05 2017

@author: Valentin Emiya, AMU & CNRS LIF
"""
import numpy as np
from data_utils import randomize_data

""" Small programme to run randomize_data """
n_examples = 5
data_dim = 4

X = np.arange(0, 20, 1).reshape((5, 4))
y = np.arange(0, -20, -4)

Xr, yr = randomize_data(X, y)

print("\n X: \n", X)
print("\n Xr: \n", Xr)
print("\n y: \n", y)
print("\n yr: \n", yr)
