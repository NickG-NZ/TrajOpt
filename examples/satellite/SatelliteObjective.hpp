/**
 * Objective for satellite attitude
 * 
 * Author: Nick Goodson
 * 
 */

#pragma once

#include "Objective.hpp"
#include "Quaternion.hpp"


class SatelliteObjective: public ObjectiveBase
{
public:

    SatelliteObjective(const double quatWeight, const Eigen::Matrix3d& omegaWeight,
                       const Eigen::Matrix3d& controlWeight, const double quatWeightTerminal,
                       const Eigen::Matrix3d& omegaWeightTerminal);

    double evaluateCost(const Eigen::VectorXd& state, const Eigen::VectorXd& goal_state,
                        const Eigen::Vector3d& control, bool terminal) override;

    double get_quat_weight() { return quatWeight_; }
    Eigen::Matrix3d get_omega_weight() { return omegaWeight_; }
                    
protected:

    double geodesic_attitude_cost(const Eigen::Vector4d& q, const Eigen::Vector4d& q_des, double& sign);

    // running cost weights
    double quatWeight_;
    Eigen::Matrix3d omegaWeight_;
    Eigen::Matrix3d controlWeight_;

    // terminal cost weights
    double quatWeightTerminal_;
    Eigen::Matrix3d omegaWeightTerminal_;

};

