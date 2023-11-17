#include "MilqrSolver.hpp"
#include "OSQPSolver.hpp"
#include "common.hpp"
#include <algorithm>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::all;

namespace trajopt {

MilqrSolver::MilqrSolver(std::shared_ptr<ObjectiveBase> objective,
                         std::shared_ptr<DynamicsBase> dynamics,
                         const SolverSettings& settings,
                         const ErrorFunction& errorFunc) :
    objective_(objective),
    dynamics_(dynamics),
    settings_(settings),
    errorFunction_(errorFunc)
{
}


// TODO: Should check if length of initialTrajectory matches horizon specified in settings
// TODO: But actually want to change this to templates so sizes will have to match
StatusCode MilqrSolver::initialize(MatrixXd initialStateTrajectory,
                                   MatrixXd initialControlSequence,
                                   VectorXd goalState)
{
    if (!objective_) {
        return StatusCode::INVALID_OBJECTIVE;
    }
    if (!dynamics_) {
        return StatusCode::INVALID_DYNAMICS;
    }
    // TODO: remove this when solver is templatized
    if (!initialStateTrajectory.cols() == settings_.horizon) {
        return StatusCode::BAD_INIT;
    }
    if (!initialControlSequence.cols() == settings_.horizon) {
        return StatusCode::BAD_INIT;
    }

    Nx_ = initialStateTrajectory.rows();
    Nu_ = initialControlSequence.rows();

    // TODO: Determine Ne by passing dummy input to user provided error function
    Ne_ = Nx_ - 1;  // error state

    lambdaMin_ = settings_.lambdaMin;
    lambdaMax_ = settings_.lambdaMax;
    lambdaScale_ = settings_.lambdaScale;

    if (Nx_ != goalState.size()){
        return StatusCode::BAD_INIT;
    }

    initialState_ = initialStateTrajectory(Eigen::all, 0);
    xtraj_ = initialStateTrajectory;
    goalState_ = goalState;

    // set up soln matrices
    N_ = settings_.horizon;
    l_ = MatrixXd::Zero(Nu_, N_ - 1);
    K_ = MatrixXd::Zero(Nu_, Ne_ * (N_ - 1));

    alphas_ = {1, 0.1, 0.001, 0.0001};

    initialized_ = true;

    return StatusCode::INIT_SUCCESS;
}


StatusCode MilqrSolver::solve(void)
{
    if (!initialized_) {
        return;
    }
    if (!objective_ || !dynamics_) {
        return;
    }

    converged_ = false;

    // initial forward rollout of trajectory
    forwardRollout(1);
    cxx_ = cxxNew_;
    cuu_ = cuuNew_;
    cx_ = cxNew_;
    cu_ = cuNew_;
    fx_ = fxNew_;
    fu_ = fuNew_;

    cost_ = costNew_;

    // main loop
    unsigned int maxIters = settings_.maxIters;
    for (int i = 0; i < maxIters; ++i){
        iterCount_++;

        // Backward pass
        bool backpassCheck = false;
        while (!backpassCheck)
        {
            bool diverged = backwardPass();
            if (diverged) {
                updateLambda(1, dLambda_);
                if (lambda_ > lambdaMax_){
                    break;
                }
                continue;
            }
            backpassCheck = true;
        }

        double cNorm = l_.lpNorm<Eigen::Infinity>();
        if ((lambda_ < settings_.lambdaTol) && (cNorm < settings_.controlTol)) {
            // complete
            converged_ = true;
            break;
        }

        // forward line-search
        bool linesearchCheck = false;
        if (backpassCheck) {
            for (auto alpha : alphas_) {
                forwardRollout(alpha);
                double expectedChange = alpha * dV_(0) + alpha * alpha * dV_(1);
                if (expectedChange < 0) {
                    cRatio_ = (costNew_ - cost_) / expectedChange;
                } else {
                    cRatio_ = -math::sgn<double>(costNew_ - cost_);
                }

                if (cRatio_ > settings_.cRatioMin) {
                    linesearchCheck = true;
                    break;
                }
            }
        }

        // update
        if (linesearchCheck) {
            updateLambda(-1, dLambda_);

            double dcost = cost_ - costNew_;
            xtraj_ = xtrajNew_;
            utraj_ = utrajNew_;
            fx_ = fxNew_;
            fu_ = fuNew_;
            cx_ = cxNew_;
            cu_ = cuNew_;
            cxx_ = cxxNew_;
            cuu_ = cuuNew_;

            cost_ = costNew_;

            if (dcost < settings_.costTol){
                converged_ = true;
                break;
            }

        } else {
            updateLambda(1, dLambda_);
            if (lambda_ > lambdaMax_) {
                converged_ = false;
                break;
            }

        }
    }

    return converged_ ? StatusCode::SOLVER_SUCCESS : StatusCode::SOLVER_FAILURE;
}


void MilqrSolver::forwardRollout(double alpha)
{
	double J = 0;
    VectorXd control_t = utraj_(Eigen::all, 0);
    dynamics_->setInitialState(initialState_);
    bool terminal = false;

    // Zero out matrices
    // cost
    cxxNew_ = MatrixXd::Zero(Ne_, Ne_ * N_);
    cuuNew_ = MatrixXd::Zero(Nu_, Nu_ * (N_ - 1));
    cxNew_ = MatrixXd::Zero(Ne_, N_);
    cuNew_ = MatrixXd::Zero(Nu_, N_ - 1);

    // dynamics
    fxNew_ = MatrixXd::Zero(Ne_, Ne_ * (N_ - 1));
    fuNew_ = MatrixXd::Zero(Ne_, Nu_ * (N_ - 1));
    xtrajNew_ = MatrixXd::Zero(Nx_, N_);
    utrajNew_ = MatrixXd::Zero(Nu_, N_-1);

    VectorXd dx = VectorXd::Zero(Ne_);

	for (int k = 0; k < N_ - 1; ++k) {

        // TODO: Make computation of dx a function passed in at run-time
        // TODO: this code is not general at all
        dx = errorFunction_(xtrajNew(all, k), xtraj_(all, k));
        // dx(seq(3, 5)) = xtrajNew_(seq(4, 6)) - xtraj_(seq(4, 6));
        // dx(seq(0, 2)) = quatError(math::Quaternion(xtrajNew_(seq(0, 3))), math::Quaternion(xtraj_(seq(0, 3))));

        utrajNew_(all, k) = utraj_(all, k) - alpha * l_(all, k) - 
                            K_(all, seq(Nx_ * k, Nx_ * (k + 1) - 1)) * dx;

        dynamics_->step(utrajNew_, settings_.timestep);
        fxNew_(all, seq(Nx_ * k, Nx_ * (k + 1) - 1)) = dynamics_->fx_;
        fuNew_(all, seq(Nu_ * k, Nu_ * (k + 1) - 1)) = dynamics_->fu_;

        J += objective_->evaluateCost(dynamics_->state_, goalState_, control_t, terminal);
        cxxNew_(all, seq(Ne_ * k, Ne_ * (k + 1) - 1)) = objective_->cxx_;
        cuuNew_(all, seq(Nu_ * k, Nu_ * (k + 1) - 1)) =  objective_->cuu_;
        cxNew_(all, k) = objective_->cx_;
        cuNew_(all, k) = objective_->cu_;

	}
    // terminal cost
    terminal = true;
    VectorXd u_temp = VectorXd::Zero(Nu_);
	J += objective_->evaluateCost(dynamics_->state_, goalState_, u_temp, terminal);
    cxxNew_(all, seq(Ne_ * (N_ - 1), Ne_ * N_ - 1)) = objective_->cxx_;
    cxNew_(all, N_ - 1) = objective_->cx_;

    costNew_ = J;
}


void MilqrSolver::backwardPass()
{
    unsigned int N = settings_.horizon;

	// Intialize matrices for optimisation
    l_ = MatrixXd::Zero(Nu_, N - 1);
    K_ = MatrixXd::Zero(Nu_, Ne_ * (N - 1)); 
    MatrixXd Qx; 
    MatrixXd Qu;
    MatrixXd Qxx; 
    MatrixXd Quu; 
    MatrixXd Qux; 

    MatrixXd Kk = MatrixXd::Zero(Nu_,Ne_);
    double result = 0;

    // Change in cost
    dV_ = Eigen::Vector2d::Zero();

    //Set cost-to-go Jacobian and Hessian equal to final costs
    VectorXd Vx = objective_->cx_(all, N-1);
    MatrixXd Vxx = objective_->cxx_(all, seq(Ne_ * (N - 1), Ne_ * N - 1));

    // Solutions to QP
    MatrixXd lk;
    MatrixXd Luu;
    std::vector<int> free_idcs;

    bool diverged = false;
    for (int k = (N - 2); k >= 0; k--) {
        
        // Define cost gradients and Hessians
        Qx = cx_(all, k) + fx_.block(all, seq(Ne_ * k, Ne_ * (k + 1) - 1)).transpose() * Vx;
        Qu = cu_(all, k) + fu_(all, seq(Nu_ * k, Nu_ * (k + 1) - 1)).transpose() * Vx;
        Qxx = cxx_(all, seq(Ne_ * k, Ne_ * (k + 1) - 1)) +
              fx_(all, seq(Ne_ * k, Ne_ * (k + 1) - 1)).transpose() * Vxx *
              fx_(all, seq(Ne_ * k, Ne_ * (k + 1) - 1));
        Quu = cuu_(all, seq(Nu_ * k, Nu_ * (k + 1) - 1)) +
              fu_(all, seq(Nu_ * k, Nu_ * (k + 1) - 1)).transpose() *
              Vxx * fu_(all, seq(Nu_ * k, Nu_ * (k + 1) - 1));
        Qux = fu_(all, seq(Nu_ * k, Nu_ * (k + 1) - 1)).transpose() * Vxx *
              fx_(all, seq(Ne_ * k, Ne_ * (k + 1) - 1));

        // Regularization
        MatrixXd QuuR = Quu + MatrixXd::Identity(Nu_, Nu_) * lambda_;

        // Solve QP
        result = boxQpSolve(QuuR, Qu, l_, lk, Kk, Luu, free_idces);

        if (result < 2){
            diverged = true;
            return;
        }

        // Update cost to go
        Vx = Qx + Kk.transpose() * Quu * lk + Kk.transpose() * Qu + Qux.transpose() * lk;
        Vxx = Qxx + Kk.transpose() * Quu * Kk + Kk.transpose() * Qux + Qux.transpose() * Kk;
        Vxx = 0.5 * (Vxx + Vxx.transpose());  // ensures symmetric hessian

        // record control cost change for convergence check
        dV_(0) += lk.transpose() * Qu;
        dV_(1) += 0.5 * lk.transpose() * Quu * lk;

        // update control vectors
        l_(all, k) = -lk;
        K_(all, seq(Ne_ * k, Ne_ * (k + 1) - 1)) = -Kk;

    }
}


void MilqrSolver::increaseLambda(double& delta)
{
    delta = std::max(lambdaScale_ * delta, lambdaMax_);
    lambda_ = std::max(lambda_ * delta, lambdaMin_);
}


void MilqrSolver::updateLambda(double direction, double& delta)
{
    if (direction == 1) {
        delta = std::max(lambdaScale_ * delta, lambdaMax_);
        lambda_ = std::max(lambda_ * delta, lambdaMin_);

    } else {
        delta = std::min(delta / lambdaScale_, 1 / lambdaScale_);
        bool weight = lambda_ > lambdaMin_;
        lambda_ = lambda_ * delta * weight;
    }
}


VectorXd MilqrSolver::defaultErrorMetric(VectorXd x, VectorXd xNominal)
{
    return x - xNominal;
}


}

