"""
Example of direct collocation trajectory optimization for a simple
pendulum swing-up problem

Solver used is from NLOPT python library

Author: Nick Goodson
"""

import nlopt
import numpy as np

from pendulum_sim import *
from gradient_check import *


def pendulum_objective(x, grad):
    """
    Goal: to invert the pendulum

    J = (xN - xg)'QN(xN - xg) + Sum[(u'u) + (x - xg)'Q(x - xg)]

    :param x: ndarray [theta1, omega1, u1, theta2, omega2, u2, ..., uN]
    :param grad: array with same size as x
    """
    # weights
    QN = np.eye(2) * 100
    Q = np.eye(2) * 10

    # goal state
    xg = np.array([np.pi, 0])

    # terminal cost
    J = (x[-3:-1] - xg) @ QN @ (x[-3:-1] - xg)

    if grad.size > 0:
        grad[-3:-1] = 2 * QN @ (x[-3:-1] - xg)

    # cumulative cost
    n_segments = x.size // 3
    for i in range(n_segments):
        idx = i * 3
        J += x[idx+2] * x[idx+2]  # u'u

        if i < n_segments - 1:
            J += (x[idx:idx+2] - xg) @ Q @ (x[idx:idx+2] - xg)

        if grad.size > 0:
            grad[idx+2] = 2 * x[idx+2]  # for u
            if i < n_segments - 1:
                grad[idx:idx+2] = 2 * Q @ (x[idx:idx+2] - xg)  # for x

    return J


def pendulum_collocation_constraints(result, x, grad, segment_times):
    """
    Computes, the collocation points and the defects.

    nlopt expects equality constraints in the form c(x) == 0

    :param result: the constraints (collocation defects) evaluated for each trajectory segment (m)
    :param x: the decision variables (n)
    :param grad: the jacobian of the constraints (mxn)
    :param segment_times: the start and end time of each segment at which to specify the dynamics
    """
    state_size = 2
    control_size = 1
    nvar = state_size + control_size
    n_segments = segment_times.size - 1
    n = x.size
    segment_durations = segment_times[1:] - segment_times[:-1]

    compute_grad = grad.size > 0

    for i in range(n_segments):
        # find nodes either side of segment
        idx1 = i * nvar
        state1 = x[idx1:idx1+state_size]
        control1 = x[idx1+state_size:idx1+nvar]  # general for u.size > 1
        time1 = segment_times[i]

        idx2 = idx1 + nvar
        state2 = x[idx2:idx2+state_size]
        control2 = x[idx2+state_size:idx2+nvar]
        time2 = segment_times[i+1]

        T = segment_durations[i]

        if compute_grad:
            f1, jac1 = pendulum_dynamics(time1, state1, control1, get_grad=True)
            f2, jac2 = pendulum_dynamics(time2, state2, control2, get_grad=True)
        else:
            f1 = pendulum_dynamics(time1, state1, control1)
            f2 = pendulum_dynamics(time2, state2, control2)

        # collocation point
        state_c = (state1 + state2) / 2 + T * (f1 - f2) / 8
        time_c = (time1 + time2) / 2
        control_c = (control1 + control2) / 2

        if compute_grad:
            fc, jac_c = pendulum_dynamics(time_c, state_c, control_c, get_grad=True)
        else:
            fc = pendulum_dynamics(time_c, state_c, control_c)

        # collocation time gradient
        state_c_dot = -3 * (state1 - state2) / (2 * T) - (f1 + f2) / 4

        defect = fc - state_c_dot  # array of length 'state_size'

        constraint_idx = i * state_size
        result[constraint_idx:constraint_idx+state_size] = defect

        if compute_grad:
            jac = np.zeros((state_size, n))

            # term 1
            dxc_dx = np.zeros((nvar, n))  # 3xn
            dxc_dx[:,idx1:idx1+nvar] = 0.5 * np.identity(3) + np.vstack((jac1, np.zeros((1, nvar)))) * (T / 8)
            dxc_dx[:,idx2:idx2+nvar] = 0.5 * np.identity(3) - np.vstack((jac2, np.zeros((1, nvar)))) * (T / 8)
            jac += jac_c @ dxc_dx

            # term 2
            val = 3 / (2 * T)
            term = np.array([[val, 0, 0],
                            [0, val, 0]])
            jac[:,idx1:idx1+nvar] += term
            jac[:,idx2:idx2+nvar] -= term

            # term3
            jac[:,idx1:idx1+nvar] += 0.25 * jac1
            jac[:,idx2:idx2+nvar] += 0.25 * jac2

            grad[constraint_idx:constraint_idx+state_size,:] = jac


def apply_variable_scaling(x, x_nominal, upper_bound):
    """
    Scales the decision variables to improve convergence of the optimization
    Scaling scheme from Hargaves and Paris 1987

    :param x: decision variables
    :param x_nominal: nominal values (eg. baseline trajectory)
    :param upper_bound: predicted upper bound on deviation
    """
    x_scaled = (x - x_nominal) / upper_bound

    return x_scaled


def undo_variable_scaling(x_scaled, x_nominal, upper_bound):
    """
    """
    x = x_scaled * upper_bound + x_nominal

    return x


def apply_time_scaling(times, upper_bound):
    """
    Scales the times used in the problem
    :param times: the segment times for a stage
    :param upper_bound: the expected upper bound on the time (known exactly for single stage problem)
    """
    return times / upper_bound


def undo_time_scaling(times_scaled, upper_bound):
    """
    """
    return times_scaled * upper_bound


def pendulum_trajopt_direct_collocation():
    """
    Optimize a pendulum swing up trajectory using direct collocation
    and an SQP solver
    """
    # pendulum problem
    n_states = 2
    n_controls = 1

    # Set up trajectory optimization problem
    initial_state = np.array([0, 0])  # theta, omega
    run_time = 3  # [s]
    segment_times = np.linspace(0, run_time, 100)
    n_segments = segment_times.size - 1
    n_decision_variables = (n_segments + 1) * (n_states + n_controls)
    n_collocation_constraints = n_segments * n_states

    # Roll out initial trajectory
    state_log, control_log, sim_times = simulate_pendulum(initial_state, np.ones(5), np.linspace(0,run_time,5), run_time, 0.01, [])
    time_idces = [np.argmin(abs(sim_times - t)) for idx, t in enumerate(segment_times)]
    initial_state_vars = np.vstack(state_log[time_idces, :])
    initial_control_vars = np.vstack(control_log[time_idces])

    initialization = np.hstack((initial_state_vars, initial_control_vars)).flatten()

    # TODO: scale variables (need to add scaling to dynamics func?)

    # optimizer
    opt = nlopt.opt(nlopt.LD_SLSQP, n_decision_variables)
    opt.set_xtol_rel(1e-5)

    # Objective
    opt.set_min_objective(pendulum_objective)

    # Bounds
    lower_bounds = np.ones(n_decision_variables) * -np.inf
    upper_bounds = np.ones(n_decision_variables) * np.inf
    lower_bounds[:n_states] = initial_state  # set the initial state
    upper_bounds[:n_states] = initial_state

    opt.set_lower_bounds(lower_bounds)
    opt.set_upper_bounds(upper_bounds)

    # Collocation constraints
    tolerance = 1e-8 * np.ones(n_collocation_constraints)  # NLOPT uses this to infer the number of constraints
    opt.add_equality_mconstraint(lambda result, x, grad: pendulum_collocation_constraints(result, x, grad, segment_times), tolerance)

    x = opt.optimize(initialization)

    minf = opt.last_optimum_value()  # recover optimal value of objective
    print(f"Optimum Objective Value: {minf:.5f}")

    # Extract solution and convert to smooth trajectory using Hermite-Simpson cubic interp.
    opt_segment_states = x.reshape(-1, 3)[:,:n_states]
    opt_segment_controls = x.reshape(-1, 3)[:,n_states:].flatten()
    interp_opt_states, interp_opt_controls, interp_opt_times = interpolate_optimized_trajectory(opt_segment_states,
                                                                                                   opt_segment_controls,
                                                                                                   segment_times, 10)
    # Simulate trajectory and compare
    time_step = 0.01
    state_log, control_log, sim_times = simulate_pendulum(initial_state, opt_segment_controls, segment_times, run_time, time_step, [])

    plot_pendulum_trajectory(state_log, control_log, sim_times, interp_opt_states, interp_opt_controls, interp_opt_times)


def interpolate_optimized_trajectory(states, controls, times, steps_per_segment):
    """
    Cubic interpolation for states
    Linear interpolation for controls
    """
    interp_states = []
    interp_controls = []
    interp_times = []
    for i in range(times.size - 1):
        t1 = times[i]
        t2 = times[i+1]
        state1 = states[i, :]
        state2 = states[i+1, :]
        control1 = controls[i]
        control2 = controls[i+1]
        f1 = pendulum_dynamics(times[i], state1, control1, [])
        f2 = pendulum_dynamics(times[i+1], state2, control2, [])

        A = np.array([[1, t1, t1 ** 2, t1 ** 3],
                      [0, 1, 2 * t1, 3 * t1 ** 2],
                      [1, t2, t2 ** 2, t2 ** 3],
                      [0, 1, 2 * t2, 3 * t2 ** 2]])

        smooth_state_segment = np.zeros((steps_per_segment, state1.size))
        subtimes = np.linspace(t1, t2, steps_per_segment)
        for j in range(state1.size):
            cubic_ceoffs = np.linalg.solve(A, np.array([state1[j], f1[j], state2[j], f2[j]]))

            smooth_state_segment[:, j] = [cubic_ceoffs @ np.array([1, t, t ** 2, t ** 3]) for t in subtimes]

        interp_states.append(smooth_state_segment)
        interp_controls.extend([(t - t1) * (control2 - control1) / (t2 - t1) + control1 for t in subtimes])
        interp_times.extend(subtimes.tolist())

    interp_states = np.vstack(interp_states)

    return interp_states, interp_controls, interp_times


def pendulum_gradient_checks():
    """
    Checks the analytical gradients for the collocation constraints,
    the objective function and the dynamics using finite differences
    """
    n_points = 5
    state_size = 2
    control_size = 1
    states = np.random.randn(n_points, state_size)
    controls = np.random.randn(n_points, control_size)
    x = np.hstack((states, controls))

    times = np.linspace(0, 10, n_points)

    # Dynamics:
    errors = gradient_check(dyn_wrapper, dyn_grad_wrapper, x, verbose=True)
    print("Dynamics gradient errors ", errors)

    # Cost
    x = x.flatten()[np.newaxis, ...]
    errors = gradient_check(lambda x_: pendulum_objective(x_, np.empty(0)), objective_grad_wrapper, x, verbose=True)
    print("Cost gradient errors ", errors)

    # Collocation constraints
    errors = gradient_check(lambda x_: collocation_wrapper(x_, times, state_size),
                            lambda x_: collocation_grad_wrapper(x_, times, state_size),
                            x, verbose=True)
    print("Collocation constraint gradient errors ", errors)


def dyn_wrapper(x):
    """
    :param x: [x1, x2, u]
    """
    xdot = pendulum_dynamics(0, x[:2], x[2])
    return xdot


def dyn_grad_wrapper(x):
    """
    :param x: [x1, x2, u]
    """
    _, grad = pendulum_dynamics(0, x[:2], x[2], get_grad=True)
    return grad


def objective_grad_wrapper(x):
    """
    :param x: [x1, x2, u1, x3, x4, u2, ...]
    """
    grad = np.zeros_like(x)
    pendulum_objective(x, grad)
    return grad


def collocation_wrapper(x, segment_times, state_size):
    """
    :param x: [x1, x2, u1, x3, x4, u2, ...]
    :param segment_times:
    """
    n_segments = segment_times.size - 1
    results = np.zeros(n_segments * state_size)
    pendulum_collocation_constraints(results, x, np.empty(0), segment_times)

    return results


def collocation_grad_wrapper(x, segment_times, state_size):
    """
    """
    n_segments = segment_times.size - 1
    grad_array = np.zeros((n_segments * state_size, x.size))
    pendulum_collocation_constraints(np.zeros(n_segments * state_size), x, grad_array, segment_times)

    return grad_array


def main():
    # pendulum_gradient_checks()
    pendulum_trajopt_direct_collocation()


if __name__ == "__main__":
    main()
