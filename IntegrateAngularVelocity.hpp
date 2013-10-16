// Copyright (c) 2013, Michael Boyle
// See LICENSE file for details

#ifndef INTEGRATEANGULARVELOCITY_HPP
#define INTEGRATEANGULARVELOCITY_HPP

#include <vector>
#include "Quaternions.hpp"

#define Quaternion_Epsilon 1.0e-14

namespace Quaternions {

  std::vector<Quaternion> FrameFromAngularVelocity(const std::vector<Quaternion>& Omega, const std::vector<double>& T);
  void FrameFromAngularVelocity(std::vector<double> (* Omega)(const double t), const double t0, const double t1, std::vector<Quaternion>& Qs, std::vector<double>& Ts);

  std::vector<Quaternion> FrameFromAngularVelocity_2D(const std::vector<Quaternion>& Omega, const std::vector<double>& T);
  void FrameFromAngularVelocity_2D(std::vector<double> (* Omega)(const double t), const double t0, const double t1, std::vector<Quaternion>& Qs, std::vector<double>& Ts);

} // namespace Quaternions

#endif // INTEGRATEANGULARVELOCITY_HPP
