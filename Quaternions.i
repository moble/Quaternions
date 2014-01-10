// -*- c++ -*-

// Copyright (c) 2013, Michael Boyle
// See LICENSE file for details

%module Quaternions

 // Quiet warnings about overloaded operators being ignored.
#pragma SWIG nowarn=362,389,401,509

%{
  #define SWIG_FILE_WITH_INIT
  #include <vector>
%}

// Slurp up the documentation
%include "docs/Quaternions_Doc.i"

///////////////////////////////////
//// Handle exceptions cleanly ////
///////////////////////////////////
%exception {
  try {
    $action;
  } catch(int i) {
    if(i==0) {
      PyErr_SetString(PyExc_RuntimeError, "Quaternions: Index out of bounds.");
    } else if(i==1) {
      PyErr_SetString(PyExc_RuntimeError, "Quaternions: Infinitely many solutions.");
    } else if(i==2) {
      PyErr_SetString(PyExc_RuntimeError, "Quaternions: Not enough points to take a derivative.");
    } else if(i==3) {
      PyErr_SetString(PyExc_RuntimeError, "Quaternions: Vector size not understood.");
    } else if(i==4) {
      PyErr_SetString(PyExc_RuntimeError, "Quaternions: Vector size inconsistent with another vector's size.");
    } else if(i==5) {
      PyErr_SetString(PyExc_RuntimeError, "Quaternions: Cannot extrapolate quaternions.");
    } else if(i==6) {
      PyErr_SetString(PyExc_RuntimeError, "Quaternions: Failed call to GSL.");
    } else if(i==7) {
      PyErr_SetString(PyExc_RuntimeError, "Quaternions: Unknown exception.");
    } else  {
      PyErr_SetString(PyExc_RuntimeError, "Unknown exception");
    }
    return NULL;
  }
}

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
#ifndef SWIGPYTHON_BUILTIN
%ignore Quaternions::Quaternion::operator=;
#endif
%include "Quaternions.hpp"
%include "IntegrateAngularVelocity.hpp"
#if defined(SWIGPYTHON_BUILTIN)
%feature("python:slot", "sq_length", functype="lenfunc") Quaternions::Quaternion::__len__;
%feature("python:slot", "mp_subscript", functype="binaryfunc") Quaternions::Quaternion::__getitem__;
%feature("python:slot", "mp_ass_subscript", functype="objobjargproc") Quaternions::Quaternion::__setitem__;
%feature("python:slot", "tp_str",  functype="reprfunc") Quaternions::Quaternion::__str__;
%feature("python:slot", "tp_repr", functype="reprfunc") Quaternions::Quaternion::__repr__;
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
  const char* __str__() {
    std::stringstream S;
    S << std::setprecision(15) << "["
      << $self->operator[](0) << ", "
      << $self->operator[](1) << ", "
      << $self->operator[](2) << ", "
      << $self->operator[](3) << "]";
    const std::string& tmp = S.str();
    const char* cstr = tmp.c_str();
    return cstr;
  }
  const char* __repr__() {
    std::stringstream S;
    S << std::setprecision(15) << "Quaternion("
      << $self->operator[](0) << ", "
      << $self->operator[](1) << ", "
      << $self->operator[](2) << ", "
      << $self->operator[](3) << ")";
    const std::string& tmp = S.str();
    const char* cstr = tmp.c_str();
    return cstr;
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
  #endif
 };

/// Add utility functions that are specific to python.  Note that
/// these are defined in the Quaternions namespace.
%insert("python") %{


%}
