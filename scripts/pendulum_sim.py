"""
Simple pendulum dynamics simulation

Author: Nick Goodson
"""


import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

G0_ = 9.8066


def pendulum_dynamics(t, x, u, w=None, get_grad=False):
    """
    Dynamics for pendulum with massless rod and ball on end

    :param t:
    :param x: state (theta, omega)
    :param u: control torque
    :param w: parameters (can be optimized)
    :param get_grad: return the jacobian [A, B] along with the state derivatives
    :return: x_dot
    """
    # Fixed Parameters
    mass = 1  # [kg]
    length = 0.2  # [m]
    damping = 0.05  # [Nms]

    # Optimizer parameters (w)
    # TODO: Try optimize one of the parameters of the problem

    x_dot = np.zeros_like(x)

    # kinematics
    x_dot[0] = x[1]

    # dynamics
    inertia = mass * length ** 2
    x_dot[1] = (u - damping * x[1]) / inertia - G0_ * np.sin(x[0]) / length

    if get_grad:
        dfdx = np.array([[0, 1, 0],
                        [- G0_ * np.cos(x[0]) / length, -damping / inertia, 1 / inertia]])  # [A, B]

        return x_dot, dfdx

    return x_dot


def simulate_pendulum(initial_state, control_inputs, control_times, run_time, time_step, params):
    """
    Simulates the pendulum system forward in time using a simple
    rk4 method.
    The control inputs are pre-determined from the trajectory optimizer and
    are interpolated

    :param initial_state: must match traj opt
    :param control_inputs: from traj opt
    :param control_times: from traj opt
    :param run_time: must match traj opt
    :param time_step: usually much smaller than traj opt
    :param params: optimized pendulum parameters
    :return:
    """
    state_log = []
    control_log = []

    control_lookup = interpolate.interp1d(control_times, control_inputs)

    sim_times = np.arange(0, run_time, time_step)

    state = initial_state

    for t in sim_times:
        control = control_lookup(t)

        state_log.append(state)
        control_log.append(control)

        state = rk4_step(pendulum_dynamics, t, state, control, time_step, params)

    state_log = np.squeeze(np.array(state_log))
    control_log =  np.squeeze(np.array(control_log))

    return state_log, control_log, sim_times


def rk4_step(dynamics_func, time, state, control, time_step, *args):
    """
    :param args: extra arguments to pass to the dynamics function
    """
    h2 = time_step / 2

    k1 = dynamics_func(time, state, control, *args)
    k2 = dynamics_func(time + h2, state + k1 * h2, control, *args)
    k3 = dynamics_func(time + h2, state + k2 * h2, control, *args)
    k4 = dynamics_func(time + time_step, state + k3 * time_step, control, *args)

    new_state = state + time_step * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return new_state


def plot_pendulum_trajectory(sim_states, sim_controls, sim_times, traj_opt_states, traj_opt_controls, traj_opt_times):
    """
    Overlays the simulated trajectory on top of the result returned by the trajectory optimizer
    Ideally these should be identical if there is no process noise added to the simulation
    """
    fig1, ax1 = plt.subplots(2, 1)
    ax1[0].set_title("Pendulum State Trajectory")
    ax1[0].plot(sim_times, np.rad2deg(sim_states[:, 0]), label="Sim")
    ax1[0].plot(traj_opt_times, np.rad2deg(traj_opt_states[:, 0]), label="Traj-opt")
    ax1[0].set(xlabel="Time [s]", ylabel="Angle [deg]")
    ax1[0].legend()
    ax1[1].plot(sim_times, sim_states[:, 1], label="Sim")
    ax1[1].plot(traj_opt_times, traj_opt_states[:, 1], label="Traj-opt")
    ax1[1].set(xlabel="Time [s]", ylabel="Angular Rate [rad/s]")
    ax1[1].legend()

    fig2, ax2 = plt.subplots()
    ax2.set_title("Control Input")
    ax2.plot(sim_times, sim_controls, label="Sim")
    ax2.plot(traj_opt_times, traj_opt_controls, label="Traj-opt")
    ax2.set(xlabel="Time [s]", ylabel="Control Torque [Nm]")
    ax2.legend()

    plt.show()


def main():

    initial_state = np.array([1, 0])

    N = 10
    opt_controls = np.zeros(N)
    opt_control_times = np.linspace(0, 10, N)
    opt_states = np.zeros((N, 2))
    params = []

    run_time = 10  # [s]
    time_step = 0.01  # [s]

    # Simulate the pendulum system using the open-loop traj opt controls
    state_log, control_log, times = simulate_pendulum(initial_state, opt_controls, opt_control_times, run_time, time_step, params)

    plot_pendulum_trajectory(state_log, control_log, times, opt_states, opt_controls, opt_control_times)


if __name__ == "__main__":
    main()
