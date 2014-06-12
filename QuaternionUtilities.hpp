#ifndef QUATERNIONUTILITIES_HPP
#define QUATERNIONUTILITIES_HPP

#include <vector>

namespace Quaternions {
  unsigned int hunt(const std::vector<double>& x, const double x_i, unsigned int i=0);
  unsigned int huntRight(const std::vector<double>& x, const double x_i, unsigned int i=0);

  std::vector<double> operator+(const std::vector<double>& a, const double b);

} // namespace Quaternions

#endif // QUATERNIONUTILITIES_HPP
