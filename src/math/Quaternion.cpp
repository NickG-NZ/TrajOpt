/**
 * Quaternion (active rotation)
 * 
 * Author: Nick Goodson
 */

#include "Quaternion.hpp"
#include <cmath>
#include <cstdio>
#include <Eigen/Dense>

using Eigen::Vector3d;
using Eigen::Matrix3d;


namespace math {

Quaternion::Quaternion() :
    s_(1.0), v1_(0.0), v2_(0.0), v3_(0.0)
{
}


Quaternion::Quaternion(double s, double v1, double v2, double v3) :
    s_(s), v1_(v1), v2_(v2), v3_(v3)
{
}


Quaternion::Quaternion(const Eigen::Vector3d& vec) :
    s_(0.0), v1_(vec(0)), v2_(vec(1)), v3_(vec(2))
{
}


double Quaternion::norm() const
{
    return sqrt(s_ * s_ + v1_ * v1_ + v2_ * v2_ + v3_ * v3_);
}


void Quaternion::normalize()
{
    double q_norm = norm();
    s_ /= q_norm;
    v1_ /= q_norm;
    v2_ /= q_norm;
    v3_ /= q_norm;
}


Quaternion Quaternion::inverse() const
{
    return Quaternion(s_, -v1_, -v2_, -v3_);
}


Vector3d Quaternion::rotate(const Eigen::Vector3d& vec)
{
    Quaternion qOut = *(this) * Quaternion(vec) * (*this).inverse();
    return qOut.getVector();
}


Vector3d Quaternion::asEulerZyx() const
{
    // derived by first finding the equivalent rotation matrix using L(q) * R(q), then solving
    // for the euler angles by comparison to matrix generated as R(x) * R(y) * R(z)
    double x = atan2(2 * (s_ * v1_ + v2_ * v3_), (s_ * s_ - v1_ * v1_ - v2_ * v2_ + v3_ * v3_));
    double y = -asin(2 * (s_ * v2_ - v1_ * v3_));
    double z = atan2(2 * (s_ * v3_ + v1_ * v2_), (s_ * s_ + v1_ * v1_ - v2_ * v2_ - v3_ * v3_));
    return Eigen::Vector3d(x, y, z);
}


Matrix3d Quaternion::asRotationMatrix() const
{
    // Can compute this matrix using the L(q) and R(q) operators. R = L(q) * R(q)
    double sp = s_ * s_;
    double v1p = v1_ * v1_;
    double v2p = v2_ * v2_;
    double v3p = v3_ * v3_;
    Eigen::Matrix3d dcm;
    dcm <<
        sp + v1p - v2p - v3p, 2 * (v1_ * v2_ - s_ * v3_), 2 * (s_ * v2_ + v1_ * v3_),
        2 * (s_ * v3_ + v1_ * v2_), sp - v1p + v2p - v3p, 2 * (v2_ * v3_ - s_ * v1_),
        2 * (v1_ * v3_ - s_ * v2_), 2 * (s_ * v1_ + v2_ * v3_), sp - v1p - v2p + v3p
    ;
    return dcm;
}


Vector3d Quaternion::asRodriguezParams() const
{
    double r1 = v1_ / s_;
    double r2 = v2_ / s_;
    double r3 = v3_ / s_;
    return Eigen::Vector3d(r1, r2, r3);
}


Quaternion& Quaternion::fromEulerZyx(const double& z, const double& y, const double& x)
{
    // Apply three active rotations to move the inertial frame into the body frame.
    // Compounding order is backwards for active rotations about moving axes (z * y * x)
    
    Quaternion z_rot(cos(z / 2), 0, 0, sin(z / 2));
    Quaternion y_rot(cos(y / 2), 0, sin(y / 2), 0);
    Quaternion x_rot(cos(x / 2), sin(x / 2), 0, 0);
    (*this) = z_rot * y_rot * x_rot;
    (*this).normalize();
    return *this;
}


Quaternion& Quaternion::fromEulerZyx(const Eigen::Vector3d& eulerZyx)
{
    return (*this).fromEulerZyx(eulerZyx(2), eulerZyx(1), eulerZyx(0));
}


Quaternion& Quaternion::fromRotationMatrix(const Eigen::Matrix3d& mat)
{
    double x = atan2(mat(2, 1), mat(2, 2));
    double y = -asin(mat(2, 0));
    double z = atan2(mat(1, 0), mat(0, 0));
    return (*this).fromEulerZyx(z, y, x);
}


Quaternion& Quaternion::fromRodriguezParams(const Eigen::Vector3d& rParams)
{
    double rParamNorm = rParams.norm();
    double scale = sqrt(1 + rParamNorm * rParamNorm);
    Quaternion q(1 / scale, rParams[0] / scale, rParams[1] / scale, rParams[2] / scale);
    (*this) = q;
    return *this;
}


Quaternion Quaternion::operator*(const Quaternion& q) const
{
    // Note sign of cross product used in derivation matches Hamilton convention 
    // In aerospace it is common to negate this to make quternions compound like DCMs
    Quaternion q_out(s_ * q.s_ - v1_ * q.v1_ - v2_ * q.v2_ - v3_ * q.v3_,
                    s_ * q.v1_ + q.s_ * v1_ + v2_ * q.v3_ - v3_ * q.v2_,
                    s_ * q.v2_ + q.s_ * v2_ - v1_ * q.v3_ + v3_ * q.v1_,
                    s_ * q.v3_ + q.s_ * v3_ + v1_ * q.v2_ - v2_ * q.v1_
    );
    return q_out;  
}


Quaternion& Quaternion::operator*=(const Quaternion& q)
{
    *this = *(this) * q;
    return *this;
}


bool operator==(const Quaternion &qLhs, const Quaternion &qRhs)
{   
    return qLhs.isEqual(qRhs);
}


bool operator!=(const Quaternion &qLhs, const Quaternion &qRhs)
{
    return !(qLhs == qRhs);
}


bool Quaternion::isEqual(const Quaternion& q) const
{
    // TODO: update this to include double cover
    printf("Warning - % - this method needs updating", __func__);
    return (s_ == q.s_) && (v1_ == q.v1_) && (v2_ == q.v2_) && (v3_ == q.v3_);
}

}