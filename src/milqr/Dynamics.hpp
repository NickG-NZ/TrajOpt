#pragma once

#include <Eigen/Dense>

namespace trajopt {

// TODO: Template this to statically allocate matrices
class DynamicsBase
{
public:
    DynamicsBase();

    /**
     * @brief 
     * 
     * @param control 
     * @param timestep 
     */
    virtual void step(const Eigen::VectorXd& control, double timestep);

    /**
     * @brief Set the Initial State object
     * 
     * @param initial_state 
     */
    virtual void setInitialState(const Eigen::VectorXd& initial_state);

protected:
    friend class MilqrSolver;

    // Store the state and jacobians for the current time-step
    Eigen::VectorXd state_;
    Eigen::VectorXd fx_;
    Eigen::MatrixXd fu_;
    
};

}
