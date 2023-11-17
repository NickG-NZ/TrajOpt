#pragma once

#include "Objective.hpp"
#include "Dynamics.hpp"
#include "math/Quaternion.hpp"
#include "StatusCode.hpp"
#include <vector>
#include <memory>
#include <functional>

namespace trajopt {

struct SolverSettings
{
    double timestep = 0.1;  // [s]
    unsigned int horizon;
    unsigned int maxIters;

    // convergence
    double costTol;
    double controlTol;
    double lambdaTol;
    double cRatioMin;

    // stability
    double lambdaMax;
    double lambdaMin;
    double lambdaScale;
};

struct SolutionStats
{

};

// TODO: Build the solver with dynamic sizing first, debug/test.
// TODO: Later, convert to static sizing with templates
// template<std::size_t N, std::size_t Nx, std::size_t Nu>
class MilqrSolver
{
public:
    MilqrSolver() = delete;

    /**
     * @param objective 
     * @param dynamics 
     * @param settings 
     */
    MilqrSolver(std::shared_ptr<ObjectiveBase> objective,
                std::shared_ptr<DynamicsBase> dynamics,
                const SolverSettings& settings,
                const ErrorFunction& errorFunc);

    /**
     * @brief     // TODO: specify all sizes as template parameters
     * 
     * @param initialStateTrajectory 
     * @param goalState 
     * @param initialControlSequence 
     * @return StatusCode 
     */
    [[nodiscard]] StatusCode initialize(Eigen::MatrixXd initialStateTrajectory,
                                        Eigen::MatrixXd initialControlSequence,
                                        Eigen::VectorXd goalState);
    
    /**
     * @brief 
     * 
     * @return StatusCode 
     */
    [[nodiscard]] StatusCode solve();

    /**
     * @brief 
     * 
     * @return true 
     * @return false 
     */
    bool solverConverged() const { return converged_; }

    // TODO: Write solution accessors...

    // TODO: Need a template parameter for error state size for static MILQR solver
    using ErrorFunction = std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)>;

protected:

    void forwardRollout(double alpha);
    bool backwardPass();
    void updateLambda(double direction, double delta);

    /**
     * @brief default error function. Assumes state space is subset of R^N and uses
     *        Euclidean error between state vectors (x - xNom)
     * @param x
     * @param xNominal
     */
    static Eigen::VectorXd defaultErrorMetric(Eigen::VectorXd x, Eigen::VectorXd xNominal);

    std::shared_ptr<ObjectiveBase> objective_;
    std::shared_ptr<DynamicsBase> dynamics_;

    SolverSettings settings_;

    ErrorFunction errorFunction_;

    bool initialized_ = false;

    // outputs
    Eigen::MatrixXd xtraj_;
    Eigen::MatrixXd utraj_;
    Eigen::MatrixXd K_;

    Eigen::VectorXd initialState_;
    Eigen::VectorXd goalState_;
    Eigen::MatrixXd l_; // feedforward corrections
    Eigen::MatrixXd xtrajNew_;
    Eigen::MatrixXd utrajNew_;

    // convergence
    bool converged_ = false;
    double expectedChange_ = 0;
    double cRatio_ = 0;
    Eigen::Vector2d dV_;
    unsigned int iterCount_ = 0;

    // regularization
    double lambda_ = 1;
    double dLambda_ = 1;
    double lambdaMax_;
    double lambdaMin_;
    double lambdaScale_;
    std::vector<double> alphas_;
    
    unsigned int Nx_;
    unsigned int Nu_;
    unsigned int Ne_;

	// cost
    double cost_;
    double costNew_;
    Eigen::MatrixXd cxx_;
    Eigen::MatrixXd cuu_;
    Eigen::MatrixXd cx_;
    Eigen::MatrixXd cu_;
    Eigen::MatrixXd cxxNew_;  // new
    Eigen::MatrixXd cuuNew_;
    Eigen::MatrixXd cxNew_;
    Eigen::MatrixXd cuNew_;

    // dynamics
    Eigen::MatrixXd fx_;
    Eigen::MatrixXd fu_;
    Eigen::MatrixXd fxNew_;  // new
    Eigen::MatrixXd fuNew_;
};

}