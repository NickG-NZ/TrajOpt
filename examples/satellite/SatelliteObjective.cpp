/**
 * Satellite Objective function
 * 
 * Author: Nick Goodson
 */

#include "SatelliteObjective.hpp"


SatelliteObjective::SatelliteObjective(const double quat_weight, const Eigen::Matrix3d& omega_weight, const Eigen::Matrix3d& control_weight,
                       const double quat_weight_terminal, const Eigen::Matrix3d& omega_weight_terminal)

: quatWeight_(quat_weight),
  omegaWeight_(omega_weight),
  controlWeight_(control_weight),
  quatWeightTerminal_(quat_weight_terminal),
  omegaWeightTerminal_(omega_weight_terminal)
{
    step_cost_ = 0;
}


double SatelliteObjective::evaluateCost(const Eigen::VectorXd& state, const Eigen::VectorXd& goal_state,
                                        const Eigen::Vector3d& control, bool terminal)
{
    // extract state
    Eigen::Vector4d q_temp = state(Eigen::seq(0, 3));
    Quaternion q(q_temp(0), q_temp(1), q_temp(2), q_temp(3));
    Eigen::Vector3d omega = state(Eigen::seq(4, 6));

    // extract goal state
    Eigen::Vector4d q_temp_g = goal_state(Eigen::seq(0, 3));
    Quaternion q_goal(q_temp_g(0), q_temp_g(1), q_temp_g(2), q_temp_g(3));
    Eigen::Vector3d omega_goal = goal_state(Eigen::seq(4, 6));

    // compute geodesic quat cost
    double sign = 0;
    double quat_cost = geodesic_attitude_cost(q_temp, q_temp_g, sign);

    Eigen::Matrix3d Qw;
    double qw;
    if (terminal){
        Qw = omegaWeightTerminal_;
        qw = quatWeightTerminal_;
    } else {
        Qw = omegaWeight_;
        qw = quatWeight_;
    }

    // Find total quadratic cost
    step_cost_ = qw * quat_cost + 0.5 * (omega - omega_goal).transpose() * Qw * (omega - omega_goal) + 
                0.5 * control.transpose() * controlWeight_ * control;

    // State hessian
    cxx_ = Eigen::MatrixXd::Zero(6, 6);
    cxx_(Eigen::seq(0, 2), Eigen::seq(0, 2)) = -Eigen::MatrixXd::Identity(3, 3) * quat_cost * sign;
    cxx_(Eigen::seq(3, 5), Eigen::seq(3, 5)) = Qw;

    // State jacobian
    cx_ = Eigen::VectorXd::Zero(6);
    Eigen::MatrixXd Gq(4, 3);
    Gq << -q.v1_, -q.v2_, -q.v3_,
          q.s_, -q.v3_, q.v2_,
          q.v3_, q.s_, -q.v1_,
          -q.v2_, q.v1_, q.s_;
           
    cx_(Eigen::seq(0, 3)) = sign * quatWeight_ * Gq.transpose() * state(Eigen::seq(0, 4));

    // Control jacobian and hessian
    cuu_ = controlWeight_;
    cu_ = controlWeight_ * control;

}

double SatelliteObjective::geodesic_attitude_cost(const Eigen::Vector4d& q, const Eigen::Vector4d& q_des, double& sign)
{
    double quat_cost = q.transpose() * q_des;

    if (1.0 + quat_cost < 1.0 - quat_cost){
        quat_cost =  1.0 + quat_cost;
        sign = 1;
    } else {
        quat_cost = (1.0 - quat_cost);
        sign = -1;
    }

    return quat_cost;
}

