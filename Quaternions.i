// -*- c++ -*-

// Copyright (c) 2014, Michael Boyle
// See LICENSE file for details

%module Quaternions

 // Quiet warnings about overloaded operators being ignored.
#pragma SWIG nowarn=362,389,401,509

#ifndef SWIGIMPORTED

%{
  #define SWIG_FILE_WITH_INIT
  #include <vector>
%}

// Slurp up the documentation
%include "docs/Quaternions_Doc.i"

///////////////////////////////////
//// Handle exceptions cleanly ////
///////////////////////////////////

// The following will appear in the header of the `_wrap.cpp` file.
%{
  const char* const QuaternionsErrors[] = {
    "This function is not yet implemented.",
    "Unknown exception",// "Failed system call.",
    "Unknown exception",// "Bad file name.",
    "Failed GSL call.",
    "Unknown exception",
    "Unknown exception",
    "Unknown exception",
    "Unknown exception",
    "Unknown exception",
    "Unknown exception",
    "Unknown exception",// "Bad value.",
    "Unknown exception",// "Bad switches; we should not have gotten here.",
    "Index out of bounds.",
    "Unknown exception",
    "Unknown exception",
    "Vector size mismatch.",
    "Unknown exception",// "Matrix size mismatch.",
    "Unknown exception",// "Matrix size is assumed to be 3x3 in this function.",
    "Not enough points to take a derivative.",
    "Unknown exception",// "Empty intersection requested.",
    "Infinitely many solutions in quaternion algebra.",
    "Vector size should be 3 or 4 in Quaternion constructor.",
    "Cannot extrapolate Quaternion arrays."
  };
  const int QuaternionsNumberOfErrors = 24;
  PyObject* const QuaternionsExceptions[] = {
    PyExc_NotImplementedError, // Not implemented
    PyExc_RuntimeError, // PyExc_SystemError, // Failed system call
    PyExc_RuntimeError, // PyExc_IOError, // Bad file name
    PyExc_RuntimeError, // GSL failed
    PyExc_RuntimeError, // [empty]
    PyExc_RuntimeError, // [empty]
    PyExc_RuntimeError, // [empty]
    PyExc_RuntimeError, // [empty]
    PyExc_RuntimeError, // [empty]
    PyExc_RuntimeError, // [empty]
    PyExc_RuntimeError, // PyExc_ValueError, // Bad value
    PyExc_RuntimeError, // PyExc_ValueError, // Bad switches
    PyExc_IndexError, // Index out of bounds
    PyExc_RuntimeError, // [empty]
    PyExc_RuntimeError, // [empty]
    PyExc_AssertionError, // Mismatched vector size
    PyExc_RuntimeError, // PyExc_AssertionError, // Mismatched matrix size
    PyExc_RuntimeError, // PyExc_AssertionError, // 3x3 matrix assumed
    PyExc_AssertionError, // Not enough points for derivative
    PyExc_RuntimeError, // PyExc_AssertionError, // Empty intersection
    PyExc_ArithmeticError, // Infinitely many solutions
    PyExc_AssertionError, // Bad vector size to constructor
    PyExc_ValueError, // Cannot extrapolate quaternions
  };
%}

// This will go inside every python wrapper for any function I've
// included; the code of the function itself will replace `$action`.
// It's a good idea to try to keep this part brief, just to cut down
// the size of the wrapper file.
%exception {
  try {
    $action;
  } catch(int i) {
    std::stringstream s;
    if(i>-1 && i<QuaternionsNumberOfErrors) { s << "Quaternions exception: " << QuaternionsErrors[i]; }
    else  { s << "Quaternions: Unknown exception number {" << i << "}"; }
    PyErr_SetString(QuaternionsExceptions[i], s.str().c_str());
    return NULL;
  }
}

#endif // SWIGIMPORTED

/////////////////////////////////////////////////
//// These will be needed by the c++ wrapper ////
/////////////////////////////////////////////////
%{
  #include <iostream>
  #include <iomanip>
  #include <complex>
  #include "Quaternions.hpp"
  #include "IntegrateAngularVelocity.hpp"
%}

%include "Quaternions_typemaps.i"

/////////////////////////////////////
//// Import the quaternion class ////
/////////////////////////////////////
%apply double& OUTPUT { double& deltat };
%apply Quaternions::Quaternion& Quaternion_argout { Quaternions::Quaternion& R_delta };
#ifndef SWIGPYTHON_BUILTIN
%ignore Quaternions::Quaternion::operator=;
#endif
%include "Quaternions.hpp"
%include "IntegrateAngularVelocity.hpp"
#if defined(SWIGPYTHON_BUILTIN)
%feature("python:slot", "mp_length", functype="lenfunc") Quaternions::Quaternion::__len__;
%feature("python:slot", "mp_subscript", functype="binaryfunc") Quaternions::Quaternion::__getitem__;
%feature("python:slot", "mp_ass_subscript", functype="objobjargproc") Quaternions::Quaternion::__setitem__;
%feature("python:slot", "tp_str",  functype="reprfunc") Quaternions::Quaternion::__str__;
%feature("python:slot", "tp_repr", functype="reprfunc") Quaternions::Quaternion::__repr__;
%feature("python:slot", "nb_divide") Quaternions::Quaternion::__div__;
%feature("python:slot", "nb_floor_divide") Quaternions::Quaternion::__div__;
%feature("python:slot", "nb_true_divide") Quaternions::Quaternion::__div__;
#endif // SWIGPYTHON_BUILTIN
%extend Quaternions::Quaternion {
  unsigned int __len__() const {
    return 4;
  }
  inline double __getitem__(const unsigned int i) const {
    return (*$self)[i];
  }
  inline void __setitem__(const unsigned int i, const double a) {
    (*$self)[i] = a;
  }
  std::string __str__() {
    std::stringstream S;
    S << std::setprecision(15) << "["
      << $self->operator[](0) << ", "
      << $self->operator[](1) << ", "
      << $self->operator[](2) << ", "
      << $self->operator[](3) << "]";
    return S.str();
  }
  std::string __repr__() {
    std::stringstream S;
    S << std::setprecision(15) << "Quaternion("
      << $self->operator[](0) << ", "
      << $self->operator[](1) << ", "
      << $self->operator[](2) << ", "
      << $self->operator[](3) << ")";
    return S.str();
  }
  #ifndef SWIGPYTHON_BUILTIN
  // This allows nicer manipulations
  %pythoncode{
    def __pow__(self, P) :
        return self.pow(P)
    __radd__ = __add__
    def __rsub__(self, t) :
        return -self+t
    __rmul__ = __mul__
    def __rdiv__(self, t) :
        return self.inverse()*t
  };
  #else
  inline Quaternion __radd__(const double a) { return (*self)+a; }
  inline Quaternion __rsub__(const double a) { return (-(*self))+a; }
  inline Quaternion __rmul__(const double a) { return (*self)*a; }
  inline Quaternion __rdiv__(const double a) { return (self->inverse())*a; }
  #endif
 };

/// Add utility functions that are specific to python.  Note that
/// these are defined in the Quaternions namespace.
%insert("python") %{


%}
