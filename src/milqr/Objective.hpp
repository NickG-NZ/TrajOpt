#pragma once

#include <Eigen/Dense>

class ObjectiveBase
{
public:
    ObjectiveBase();

    /**
     * @brief 
     * 
     * @param state 
     * @param goal_state 
     * @param control 
     * @param terminal 
     * @return double 
     */
    virtual double evaluateCost(const Eigen::VectorXd& state, const Eigen::VectorXd& goal_state,
                                 const Eigen::Vector3d& control, bool terminal);

protected:
    friend class MilqrSolver;

    double stepCost_;
    Eigen::VectorXd cx_;  // state jacobian
    Eigen::VectorXd cu_;  // control jacobian
    Eigen::MatrixXd cxx_; // state hessian
    Eigen::MatrixXd cuu_; // control hessian 

};


