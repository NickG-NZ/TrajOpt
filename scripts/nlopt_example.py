"""
Solving example non-linear optimization problem with the NLOPT
python library

minimize sqrt(x(2))

subject to   x(2) >= (2x(1) + 0)^3
             x(2) >= (-x(1) + 1)^3
             x(2) >= 0


Optimum at x(1) = 1/3
           x(2) = 8/27
"""

import nlopt
import numpy as np


def objective(x, grad):
    """
    The non-linear objective

    :param x: the decision variables
    :param grad: if non empty, this must be modified in place to the gradient of the objective
    :return: the value of the objective
    """
    if grad.size > 0:
        grad[0] = 0
        grad[1] = 0.5 / np.sqrt(x[1])

    return np.sqrt(x[1])


def constraint(x, grad, a, b):
    """
    A single non-linear constraint. For this particular problem,
    this same constraint interface can handle both NL constraints.
    Other problems would require more than one function to be defined.

    :param x: the decision variables
    :param grad: if non empty, this must be modified in place to the gradient of the constraint

    Additional arguments that are not part of the default interface
    :param a:
    :param b:
    :return: the value of the constraints
    """

    # NLOPT expects constraints in the form g(x) <= 0
    if grad.size > 0:
        grad[0] = 3 * a * (a * x[0] + b) ** 2
        grad[1] = -1.0

    return (a * x[0] + b) ** 3 - x[1]


def run_nlopt_example():
    """
    Example problem
    """
    opt = nlopt.opt(nlopt.LD_MMA, 2)  # (solver, num decision vars)
    opt.set_lower_bounds([-float('inf'), 0])

    opt.set_min_objective(objective)

    # Use a lambda to pass extra arguments to the constraint function
    opt.add_inequality_constraint(lambda x, grad: constraint(x, grad, 2, 0), 1e-8)  # (func, tolerance)
    opt.add_inequality_constraint(lambda x, grad: constraint(x, grad, -1, 1), 1e-8)

    opt.set_xtol_rel(1e-4)

    x = opt.optimize([1.234, 5.678])  # (initial guess)

    minf = opt.last_optimum_value()  # recover optimal value of objective

    print(f"Optimum Value: {minf:.5f}")
    print(f"Occurs at: {x[0]:.5f}, {x[1]:.5f}")
    print(f"Result code: {opt.last_optimize_result()}")  # don't need to check this as exception will be thrown


def main():
    """
    Run the optimization
    """
    run_nlopt_example()


if __name__ == "__main__":
    main()



