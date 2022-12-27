#!/usr/bin/env python3.9

"""Histogram: number of songs per release year"""

import numpy as np
import matplotlib.pyplot as plt

from data_utils import load_data


_, y_labeled, _ = load_data("data/YearPredictionMSD_100.npz")

fig = plt.figure()
ax = plt.axes()
plt.hist(y_labeled,
         bins=np.arange(int(min(y_labeled)), int(max(y_labeled)), 1))
plt.style.context("seaborn")
plt.xlabel("Release year")
plt.ylabel("Number of songs")
plt.title("Histogram of the number of songs per release year"
          + "\n from the MSD_100 dataset")
plt.savefig("figures/hist_year.png")
