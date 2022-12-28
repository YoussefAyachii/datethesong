#!/usr/bin/env python3.9

"""Histogram: number of songs per release year"""

import numpy as np
import matplotlib.pyplot as plt

from data_utils import load_data


_, y_labeled, _ = load_data("data/YearPredictionMSD_100.npz")

fig = plt.figure()
ax = plt.axes()
plt.hist(y_labeled, bins=np.arange((int(min(y_labeled))),
                                    int(max(y_labeled))+1, 1))
plt.style.context("seaborn")
plt.xlabel("Release year")
plt.ylabel("Number of songs")
plt.title("Histogram of the number of songs per release year"
          + "\n from the MSD_100 dataset")
plt.savefig("figures/hist_year.png")

# more info on data set
print("number of songs in the training set:\n", len(y_labeled))

print("average number of songs released in or after 1950"
      + "in X_labeled (training set) \n",
      np.round(np.sum(y_labeled>=1950) / (max(y_labeled)-1949), 2))

print("average number of songs released before 1950"
      + "in X_labeled (training set) \n",
      np.round(np.sum(y_labeled<1950) / (max(y_labeled)-1949), 2))