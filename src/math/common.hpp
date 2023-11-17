#pragma once

#include <Eigen/Dense>
#include <Quaternion.hpp>

namespace math {

/**
 * @brief 
 * 
 * @tparam T 
 * @param val 
 * @return int 
 */
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}


Eigen::Matrix3d crossProductMatrix(const Eigen::Vector3d& vec)
{
    Eigen::Matrix3d mat;
    mat << 0,    -vec(2), vec(1),
          vec(2),   0,    vec(0),
          -vec(1), -vec(0),  0;

    return mat;

}


/**
 * @brief Compute the error quaternion representing the difference between two rotations
 *        The error is computed such that q = qNominal * qErr
 *        (ie. the quaternion q is the quaternion qNom rotated by the quaternion qErr)
 * 
 * @param q 
 * @param qNominal
 * @return Eigen::Vector3d 
 */
Quaternion quaterionError(const Quaternion& q, const Quaternion& qNominal)
{
    return qNominal.inverse() * q;
}


/**
 * @brief Compute the rotation error between two quaternions and return a 3 parameter
 *        representation of the result (RodriguezParams)
 * 
 * @param q New quaternion 
 * @param qNominal Base quaternion
 * @return Eigen::Vector3d Rodriguez parameters of error
 */
Eigen::Vector3d rotationError3D(const Quaternion& q, const Quaternion& qNominal)
{
    return quaternionError(q, qNomainl).asRodriguezParams();
}



}

