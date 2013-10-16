#include <vector>
#include <iostream>
#include <iomanip>
#include "Quaternions.hpp"
#include "IntegrateAngularVelocity.hpp"
using namespace std;
using namespace Quaternions;

const double alpha = 0.05;
const double T_i = 0.0;
const double T_f = 20000.0;

Quaternion R_1func(const double t) {
  return Quaternions::exp(alpha*t*Quaternions::xHat/2.)
    *Quaternions::exp((alpha*pow(t,1.2))*Quaternions::yHat/2.)
    *Quaternions::exp((alpha*pow(t,1.3))*Quaternions::zHat/2.);
}

vector<double> Omega(const double t) {
  const Quaternion R_1 = R_1func(t);
  const Quaternion Rdot_1 = (alpha/2.0) *
    (
     Quaternions::xHat*
     Quaternions::exp(alpha*t*Quaternions::xHat/2.)
     *Quaternions::exp((alpha*pow(t,1.2))*Quaternions::yHat/2.)
     *Quaternions::exp((alpha*pow(t,1.3))*Quaternions::zHat/2.)
     +
     Quaternions::exp(alpha*t*Quaternions::xHat/2.)
     *1.2*pow(t,0.2)*Quaternions::yHat
     *Quaternions::exp((alpha*pow(t,1.2))*Quaternions::yHat/2.)
     *Quaternions::exp((alpha*pow(t,1.3))*Quaternions::zHat/2.)
     +
     Quaternions::exp(alpha*t*Quaternions::xHat/2.)
     *Quaternions::exp((alpha*pow(t,1.2))*Quaternions::yHat/2.)
     *1.3*pow(t,0.3)*Quaternions::zHat
     *Quaternions::exp((alpha*pow(t,1.3))*Quaternions::zHat/2.)
     );
  return (2*Rdot_1*Quaternions::conjugate(R_1)).vec();
}

int main() {
  vector<Quaternion> Qs;
  vector<double> Ts;
  cout << setprecision(16);
  FrameFromAngularVelocity_2D(&Omega, T_i, T_f, Qs, Ts);
  double MaxError = 0.0;
  for(unsigned int i=0; i<Ts.size(); ++i) {
    const Quaternion R_1 = R_1func(Ts[i]);
    const Quaternion z_1 = R_1*zHat*R_1.conjugate();
    const Quaternion z_2 = Qs[i]*zHat*Qs[i].conjugate();
    const double Error = (z_1-z_2).abs();
    MaxError = (Error>MaxError ? Error : MaxError);
    cout << Ts[i]
  	 // << "\t" << Qs[i]
  	 // << "\t" << Quaternion(Omega(Ts[i]))
  	 << "\t" << Error
  	 << endl;
  }
  cerr << "Took " << Ts.size() << " steps.\tMaxError = " << MaxError << endl;

  return 0;
}
