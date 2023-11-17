/**
 * Satellite implementation
 * 
 * Author: Nick Goodson
 */

#include "SatelliteDynamics.hpp"
#include "math/common.hpp"
#include <cmath>

using Eigen::Vector3d;
using Eigen::Matrix3d;
using Eigen::VectorXd;
using Eigen::MatrixXd;


SatelliteDynamics::SatelliteDynamics(const Matrix3d& inertia, const Vector3d& control_lb,
                                     const Vector3d& control_ub)
:
    inertia_(inertia),
    controlLB_(control_lb),
    controlUB_(control_ub)
{
    inertiaInv_ = inertia.inverse();
}


void SatelliteDynamics::step(const Eigen::VectorXd& control, double time_step)
{
    if (earthMagFieldEci_.size() == 0){
        // TODO: log warning
        return;
    }

    // Step satellite dynamics (using RK2 method)
    Eigen::VectorXd xdot1(Nx_), xdot2(Nx_);  // state derivatives
    Eigen::MatrixXd dxdot1(Nx_, Nx_ + Nu_), dxdot2(Nx_, Nx_ + Nu_);  // Cts time Jacobians [A, B]

    evaluateDynamics(q_b2i, omega_b, control, xdot1, dxdot1);
    Eigen::VectorXd state_half = state_ + 0.5 * time_step * xdot1;
    math::Quaternion q_b2i_half(state_half(Eigen::seq(0, 3))); // TODO: ?? Why 0-3 (not 0-4)

    evaluateDynamics(q_b2i_half, state_half(Eigen::seq(4, Eigen::last)), control, xdot2, dxdot2);
    Eigen::VectorXd state_new = state_ + time_step * xdot2;
    q_b2i = math::Quaternion(state_new(Eigen::seq(0, 3)));
    omega_b = state_new(Eigen::seq(4, Eigen::last));    

    // Re-normalize quaternion
    q_b2i.normalize();

    // Form multiplicative attitude Jacobians
    Eigen::MatrixXd E0(7, 6);
    Eigen::MatrixXd E1(7, 6);
    E0 << -state_(1), -state_(2), -state_(3), 0, 0, 0,
           state_(0), -state_(3), state_(2), 0, 0, 0,
           state_(3), state_(0), -state_(1), 0, 0, 0,
           -state_(2), state_(1), state_(0), 0, 0, 0,
           0, 0, 0, 1, 0, 0,
           0, 0, 0, 0, 1, 0,
           0, 0, 0, 0, 0, 1;
    
    E1 << -state_new(1), -state_new(2), -state_new(3), 0, 0, 0,
           state_new(0), -state_new(3), state_new(2), 0, 0, 0,
           state_new(3), state_new(0), -state_new(1), 0, 0, 0,
           -state_new(2), state_new(1), state_new(0), 0, 0, 0,
           0, 0, 0, 1, 0, 0,
           0, 0, 0, 0, 1, 0,
           0, 0, 0, 0, 0, 1;

    // Update discrete Jacobians
    Eigen::MatrixXd A1 = dxdot1(Eigen::all, Eigen::seq(0, Nx_ - 1));
    Eigen::MatrixXd B1 = dxdot1(Eigen::all, Eigen::seq(Nx_, Eigen::last));
    Eigen::MatrixXd A2 = dxdot2(Eigen::all, Eigen::seq(0, Nx_ - 1));
    Eigen::MatrixXd B2 = dxdot2(Eigen::all, Eigen::seq(Nx_, Eigen::last));
    fx_ = E1.transpose() * (Eigen::MatrixXd::Identity(Nx_, Nx_) + time_step * A2 + 
                                    0.5 * time_step * time_step * A2 * A1) * E0;
    fu_ = E1.transpose() * (time_step * B2 * 0.5 * time_step * time_step * A2 * B1);

    // Update the state
    state_ = state_new;
    omega_b = state_new(Eigen::seq(4, Eigen::last));

}


void SatelliteDynamics::setInitialState(const Eigen::VectorXd& initial_state)
{
    state_ = initial_state;
    Eigen::Vector4d q = initial_state(Eigen::seq(0, 3)); // ??
    q_b2i = math::Quaternion(q(0), q(1), q(2), q(3));  // ??
    omega_b = initial_state(Eigen::seq(4, 6));

}


void SatelliteDynamics::updateEarthMagField(const Eigen::Vector3d& mag_field_eci)
{
    earthMagFieldEci_ = mag_field_eci;
}


void SatelliteDynamics::evaluateDynamics(const math::Quaternion& q, const Vector3d& omega,
                                         const Vector3d& u, VectorXd& xdot, MatrixXd& dxdot)
{

    // transform magnetic field into body frame
    Eigen::Vector3d mag_field_b = q_b2i.rotate(earthMagFieldEci_);

    // Quaternion kinematics
    math::Quaternion q_dot = q_b2i * math::Quaternion(omega_b);
    xdot(Eigen::seq(0, 3)) = q_dot.getVector();

    // attitude dynamics
    xdot(Eigen::seq(4, Eigen::last)) = inertiaInv_ * (-mag_field_b.cross(u) - omega_b.cross(inertia_ * omega_b));

    // Jacobians
    MatrixXd A = MatrixXd::Zero(7, 7);
    A(0, Eigen::seq(1, 3)) = -omega_b.transpose();
    A(0, Eigen::seq(4, Eigen::last)) = -q.getVector().transpose();
    A(Eigen::seq(1, 3), 0) = omega_b;
    A(Eigen::seq(1, 3), Eigen::seq(1, 3)) = -math::crossProductMatrix(omega_b);

    A(Eigen::seq(1, 3), Eigen::seq(4, 6)) = q_b2i.getScalar() * Eigen::MatrixXd::Identity(3, 3) +
                                            math::crossProductMatrix(q_b2i.getVector());

    A(Eigen::seq(4, 6), Eigen::seq(4, 6)) = -2 * inertiaInv_ * (math::crossProductMatrix(omega_b) *
                                            inertia_ - math::crossProductMatrix(inertia_ * omega_b));

    MatrixXd B = MatrixXd::Zero(7, 3);
    B(Eigen::seq(4, 6), Eigen::all) = -inertiaInv_ * math::crossProductMatrix(mag_field_b);

    dxdot(Eigen::all, Eigen::seq(0, 6)) = A;
    dxdot(Eigen::all, Eigen::seq(6, Eigen::last)) = B;

}

