from sphconvolve import convolve_positions
import matplotlib.pyplot as plt
import numpy as np


def test_integration():
    x = np.arange(0, 1000)
    y = np.array([0.0]*500 + [1.0]*500)

    y_sm = convolve_positions(y, 12, N_neigh=48, dim=1)

    plt.plot(x, y, label="Original")
    plt.plot(x, y_sm, label="Smoothed")

    plt.xlim(450, 550)

    plt.savefig("test_integration.png")
