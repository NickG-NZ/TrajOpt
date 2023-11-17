/**
 * Holds the Satellite dynamics
 * 
 * Author: Nick Goodson
 */

#pragma once

#include "Dynamics.hpp"
#include "math/Quaternion.hpp"


class SatelliteDynamics: public trajopt::DynamicsBase
{
public:
    SatelliteDynamics(const Eigen::Matrix3d& inertia, const Eigen::Vector3d& controlLB,
                      const Eigen::Vector3d& controlUB);

    void step(const Eigen::VectorXd& control, double timeStep) override;

    void setInitialState(const Eigen::VectorXd& initialState) override;

    void updateEarthMagField(const Eigen::Vector3d& magFieldEci);

    // getters
    Eigen::Matrix3d get_inertia() const { return inertia_; }
    Eigen::Vector3d get_control_lb() const { return controlLB_; }
    Eigen::Vector3d get_control_ub() const { return controlUB_; }
    unsigned int get_state_size() const { return Nx_; }
    unsigned int get_control_size() const { return Nu_; }

    // TODO: THese should not be public
    math::Quaternion q_b2i;
    Eigen::Vector3d omega_b;

protected:

    void evaluateDynamics(const math::Quaternion& q, const Eigen::Vector3d& omega,
                          const Eigen::Vector3d& u, Eigen::VectorXd& xdot, Eigen::MatrixXd& dxdot);

    // convenience vars
    const unsigned int Nx_ = 7;
    const unsigned int Nu_ = 3;

    // Satellite parameters
    Eigen::Matrix3d inertia_;
    Eigen::Matrix3d inertiaInv_; // inverse
    Eigen::Vector3d controlLB_;
    Eigen::Vector3d controlUB_;

    // Mag field
    Eigen::Vector3d earthMagFieldEci_;

};