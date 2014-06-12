#include "QuaternionUtilities.hpp"
#include <iostream>
#include "Errors.hpp"

#ifndef DOXYGEN
/// Find the largest index i with x[i]<=x_i
void hunt_in_place(const std::vector<double>& x, const double x_i, unsigned int& i) {
  /// Based on the Numerical Recipes routine of the same name
  const unsigned int xSize=x.size();
  if (xSize==0) {
    std::cerr << "\n\n" << __FILE__ << ":" << __LINE__ << ": You can't `hunt` in an empty array!" << std::endl;
    throw(IndexOutOfBounds);
  }
  if(x.back()<=x_i) {
    i = x.size()-1;
    return;
  }
  if(xSize<3) { i=0; return; }
  i = std::min(i, xSize-1);
  unsigned int i_lower=i, i_middle, i_upper, increment=1;
  bool ascend=(x[xSize-1] >= x[0]);
  if (i_lower > xSize-1) {
    i_lower=0;
    i_upper=xSize-1;
  } else {
    if ((x_i >= x[i_lower]) == ascend) {
      for (;;) {
        i_upper = i_lower + increment;
        if (i_upper >= xSize-1) { i_upper = xSize-1; break;}
        else if ((x_i < x[i_upper]) == ascend) break;
        else {
          i_lower = i_upper;
          increment += increment;
        }
      }
    } else {
      i_upper = i_lower;
      for (;;) {
        if(i_lower<=increment) { i_lower = 0; break; }
        i_lower = i_lower - increment;
        if ((x_i >= x[i_lower]) == ascend) break;
        else {
          i_upper = i_lower;
          increment += increment;
        }
      }
    }
  }
  while (i_upper-i_lower > 1) {
    i_middle = (i_upper+i_lower) >> 1;
    if ((x_i >= x[i_middle]) == ascend)
      i_lower=i_middle;
    else
      i_upper=i_middle;
  }
  i = std::min(xSize-1,i_lower);
  return;
}
#endif // DOXYGEN

/// Find the largest index i with x[i]<=x_i
unsigned int Quaternions::hunt(const std::vector<double>& x, const double x_i, unsigned int i) {
  /// Based on the Numerical Recipes routine of the same name
  hunt_in_place(x, x_i, i);
  // std::cerr << "hunt([" << x[0] << ", ..., " << x.back() << "], " << x_i << ") = " << i << std::endl;
  return i;
}

/// Find the largest index i with x[i-1]<=x_i
unsigned int Quaternions::huntRight(const std::vector<double>& x, const double x_i, unsigned int i) {
  /// Based on the Numerical Recipes routine of the same name
  hunt_in_place(x, x_i, i);
  if(i<x.size()) ++i;
  return i;
}


std::vector<double> Quaternions::operator+(const std::vector<double>& a, const double b) {
  const unsigned int size = a.size();
  std::vector<double> c(a);
  for(unsigned int i=0; i<size; ++i) {
    c[i] += b;
  }
  return c;
}
