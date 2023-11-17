"""
Checks analytical gradients using finite differences

Author: Nick Goodson
"""

import numpy as np


def gradient_check(func, grad_func, x, order=2, step_size=None, verbose=True):
    """
    :param func: function whose gradient is being evaluated
    :param grad_func: gradient function to check
    :param x: points at which to evaluate the gradient (1st dim is N points)
    :param order: the order of the numerical apporimxation (1st or 2nd)
    :param step_size: automatically selected if not provided
    """
    if len(x.shape) < 2:
        raise ValueError("invalid shape for inputs x")

    n_points = x.shape[0]
    errors = np.zeros(n_points)

    for i in range(n_points):
        grad_to_test = grad_func(x[i, ...])
        grad_fd = finite_difference_gradient(func, x[i, ...], order=order, step_size=step_size)

        if verbose:
            print(f"{i} function grad:{grad_to_test}")
            print(f"{i} numerical grad:{grad_fd}\n")

        errors[i] = np.linalg.norm(grad_to_test - grad_fd)

    return errors


def finite_difference_gradient(func, x, order=2, step_size=None):
    """
    Compute the gradient at a point using a finite difference approximation.
    Can handle scalar and vector valued functions

    The automatically selected step sizes are the same as those used by
    Matlab's fmincon
    """
    eps = np.finfo(float).eps
    c_val = func(x)
    try:
        input_dim = x.size
    except AttributeError:
        input_dim = 1  # scalar

    try:
        output_dim = c_val.size
    except AttributeError:
        output_dim = 1  # scalar

    grad = np.zeros((output_dim, input_dim))

    if order == 1:
        if not step_size:
            step_size = np.sqrt(eps)

        delta = np.eye(input_dim) * step_size
        for i in range(input_dim):
            grad[:, i] = (func(x + delta[:, i]) - c_val) / step_size

    elif order == 2:  # central difference
        if not step_size:
            step_size = eps ** (1 / 3)

        delta = np.eye(input_dim) * step_size
        for i in range(input_dim):
            grad[:, i] = (func(x + delta[:, i]) - func(x - delta[:, i])) / (2 * step_size)

    else:
        raise ValueError("Finite difference order must be either 1 or 2")

    return grad.squeeze()
