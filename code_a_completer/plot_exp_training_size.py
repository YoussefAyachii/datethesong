"""_summary_"""

import numpy as np
import matplotlib.pyplot as plt

# Load npz file
npz_ = np.load("exp_training_size_results.npz")
valid_error, train_error, N = npz_["valid_error"], npz_["train_error"], npz_["N"]
learning_time = npz_["learning_time"]

# Part 1: MSE of each linear regression method depending on the sample size
fig = plt.figure()
ax = plt.axes()

ax.semilogy(N, valid_error[:, 0], color="red", linestyle="-", label="majority")
ax.semilogy(N, valid_error[:, 1], color="blue", linestyle="-", label="median")
ax.semilogy(N, valid_error[:, 2], color="green", linestyle="-", label="mean")
ax.semilogy(N, valid_error[:, 3], color="black", linestyle="-", label="least square")

ax.semilogy(N, train_error[:, 0], color="red", linestyle="dotted", label="majority")
ax.semilogy(N, train_error[:, 1], color="blue", linestyle="dotted", label="median")
ax.semilogy(N, train_error[:, 2], color="green", linestyle="dotted", label="mean")
ax.semilogy(N, train_error[:, 3], color="black", linestyle="dotted", label="least square")

ax.set_xlabel("Sample size (N)")
ax.set_ylabel("Error (MSE)")
ax.set_title("Performance of different Linear regression strategies \n depending on the sample size"+
             " of the data set")
ax.legend()

plt.savefig("figures/plot_exp_training_size_error.png")

# Part 2 : learning time of each linear regression method depending on the sample size
fig = plt.figure()
ax = plt.axes()
ax.plot(N, learning_time[:, 0], color="red", linestyle="-", label="majority")
ax.plot(N, learning_time[:, 1], color="blue", linestyle="-", label="median")
ax.plot(N, learning_time[:, 2], color="green", linestyle="-", label="mean")
ax.plot(N, learning_time[:, 3], color="black", linestyle="-", label="least square")
plt.show()
plt.savefig("figures/plot_exp_training_size_time.png")
