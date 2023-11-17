#include "SatelliteDynamics.hpp"
#include "SatelliteObjective.hpp"
#include "src/milqr/MilqrSolver.hpp"  // TODO: make CMake install to 'kiwiOpt' instead of 'src'
#include <memory>
#include <Eigen/Dense>

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Eigen::Vector


int main(int argc, char* argv[]) {

    constexpr int Nx = 7;  // [q, omega]
    constexpr int Nu = 3;  // 

    // dynamics
    const Matrix3d inertia = Matrix3d::Identity();
    const Vector3d controlLB = -1 * Vector3d::Ones();
    const Vector3d controlUB = Vector3d::Ones();

    auto dynamics = std::make_shared<SatelliteDynamics>(inertia, controlLB, controlUB);

    // Objective
    const double quatWeight = 10;
    const double quatWeightTerminal = 10 * quatWeight;
    const Matrix3d omegaWeight = 2 * Matrix3d::Identity();
    const Matrix3d omegaWeightTerminal = 10 * omegaWeight;
    const Matrix3d controlWeight = 0.1 * Matrix3d::Identity();
    
    auto objective = std::make_shared<SatelliteObjective>(quatWeight, omegaWeight, controlWeight,
                                                          quatWeightTerminal, omegaWeightTerminal);

    // Settings
    trajopt::SolverSettings settings {
        .timestep = 0.1,
        .horizon = 200,
        .maxIters = 10,
        .costTol = 0.1,
        .controlTol = 0.1,
        .lambdaTol = 0.1,
        .cRatioMin = 0.8,
        .lambdaMax = 2,
        .lambdaMin = 0.2,
        .lambdaScale = 0.1
    };

    trajopt::MilqrSolver solver(objective, dynamics, settings, trajopt::MilqrSolver::defaultErrorMetric);

    // Initialize problem
    MatrixXd initialTraj = MatrixXd::Zero(Nx, settings.horizon);
    intialTraj.row(0).setValue(1);  // identity quaternion

    trajopt::StatusCode res = solver.initialize(initialTraj, initialControls, goal);

    trajopt::StatusCode res = solver.solve();

    return 0;
}