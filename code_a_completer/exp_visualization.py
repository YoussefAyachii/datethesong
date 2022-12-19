#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Valentin Emiya, AMU & CNRS LIS
"""
import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_data


""" Build the histogram of the years of the songs from the training set and
export the figure to the image file hist_train.png
"""

_, y_labeled, _ = load_data("data/YearPredictionMSD_100.npz")

fig = plt.figure()
ax = plt.axes()
plt.hist(y_labeled,
         bins=np.arange(int(min(y_labeled)), int(max(y_labeled)), 1))
plt.style.context("seaborn")
plt.title("histogram of the number of songs per year")
plt.savefig("figures/hist_year.png")
