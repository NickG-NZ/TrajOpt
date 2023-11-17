/**
 * Class for representing active rotations using unit quaternions
 * 
 * Author: Nick Goodson
 */

#pragma once

#include <Eigen/Dense>

namespace math {

class Quaternion
{
public:

    /**
     * @brief Defualt initialize to identity quaternion
     * 
     */
    Quaternion();

    /**
     * @brief From known parameters
     * 
     * @param s 
     * @param v1 
     * @param v2 
     * @param v3 
     */
    Quaternion(const double s, const double v1, const double v2, const double v3);

    /**
     * @brief Convert a vector to a quaternion for multiplication purposes
     * 
     * @param vec 
     */
    explicit Quaternion(const Eigen::Vector3d& vec);

    /**
     * @brief 
     * 
     * @return double 
     */
    double norm() const;

    /**
     * @brief 
     * 
     */
    void normalize();

    /**
     * @brief 
     * 
     * @return Quaternion 
     */
    Quaternion inverse() const;

    /**
     * @brief 
     * 
     * @param vec 
     * @return Eigen::Vector3d 
     */
    Eigen::Vector3d rotate(const Eigen::Vector3d& vec);

    /**
     * @brief Get the Vector parameter
     * 
     * @return Eigen::Vector3d 
     */
    Eigen::Vector3d getVector() const { return Eigen::Vector3d(v1_, v2_, v3_); }

    /**
     * @brief Get the Scalar parameter
     * 
     * @return double 
     */
    double getScalar() const { return s_; }

    /**
     * @brief 
     * 
     * @return Eigen::Vector3d 
     */
    Eigen::Vector3d asEulerZyx() const;

    /**
     * @brief Obtain the rotation matrix corresponding to the active quaternion
     * This is the transpose of the common aerospace DCM
     * 
     * @return Eigen::Matrix3d 
     */
    Eigen::Matrix3d asRotationMatrix() const;

    /**
     * @brief Convert to Rodriguez params, defined from quaternion using the Cayley Map:
     * phi = q_v / q_s = r*sin(theta) / cos(theta) = r * tan(theta)
     * 
     * @return Eigen::Vector3d 
     */
    Eigen::Vector3d asRodriguezParams() const;

    // Alternative initalizers

    /**
     * @brief From z-y-x Euler angles
     * 
     * @param z 
     * @param y 
     * @param x 
     * @return Quaternion& 
     */
    Quaternion& fromEulerZyx(const double& z, const double& y, const double& x);

    /**
     * @brief Expects euler angles in alpabetical order Vector3d(x, y, z)
     * 
     * @param eulerZyx 
     * @return Quaternion& 
     */
    Quaternion& fromEulerZyx(const Eigen::Vector3d& eulerZyx);

    /**
     * @brief Expects a rotation matrix for an activate rotation (body -> inertial)
     * 
     * @param mat 
     * @return Quaternion& 
     */
    Quaternion& fromRotationMatrix(const Eigen::Matrix3d& mat);

    /**
     * @brief Inverse Cayley map
     * 
     * @param rParams 
     * @return Quaternion& 
     */
    Quaternion& fromRodriguezParams(const Eigen::Vector3d& rParams);

    /**
     * @brief Applies Hamilton convention for multiplication
     * 
     * @param q 
     * @return Quaternion 
     */
    Quaternion operator*(const Quaternion &q) const;
    Quaternion& operator*=(const Quaternion &q);
    friend bool operator==(const Quaternion& qLhs, const Quaternion& qRhs);
    friend bool operator!=(const Quaternion& qLhs, const Quaternion& qRhs);
    virtual bool isEqual(const Quaternion& q) const;

    double s_;
    double v1_;
    double v2_;
    double v3_;

protected:

};

bool operator==(const Quaternion& qLhs, const Quaternion& qRhs);
bool operator!=(const Quaternion& qLhs, const Quaternion& qRhs);

}