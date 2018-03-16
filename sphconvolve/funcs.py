"""
Main functions for the convolution code.
"""

import numpy as np

from numba import vectorize, float64

@vectorize([float64(float64)])
def cubic_spline(q):
    """
    Cubic spline kernel
    
    Inputs
    ------

    q | float | r / h for the particle.
    """

    if q <= 1:
        return 1 - 1.5 * q * q * (1 - 0.5 * q)
    elif q <=2 :
        return 0.25 * (2 - q)**3
    else:
        return 0.


def cubic_spline_discrete(h, N_neigh, dim=3):
    """
    Get the cubic spline kernel (discrete).

    Inputs
    ------

    h | float / np.ndarray | Smoothing lengths

    N_neigh | int | the number of neighbors for the kernel.

    dim | int | the number of dimensions.
    """

    q = np.linspace(-2, 2, N_neigh+1)
    kernels = cubic_spline(abs(q))

    if dim == 1:
        sigma = 2.0 / (3.0 * h)
    elif dim == 2:
        sigma = 10.0 / (7.0 * np.pi * h * h)
    elif dim == 3:
        sigma = 1.0 / (np.pi * h * h * h)

    return kernels * sigma


def convolve_positions(y, h, N_neigh=48, dim=3):
    """
    Convolve the values with the kernels.
    """

    kernel = cubic_spline_discrete(h, N_neigh, dim)

    y_left = np.ones(N_neigh) * y[0]
    y_right = np.ones(N_neigh) * y[-1]

    y_all = np.hstack([y_left, y, y_right])

    convolved = np.convolve(y_all, kernel)

    # We get some elements on the L/R that are 'spurious' in this context.
    lr = int(N_neigh / 2) + N_neigh

    return convolved[lr:-lr]


