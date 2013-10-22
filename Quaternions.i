// -*- c++ -*-

// Copyright (c) 2013, Michael Boyle
// See LICENSE file for details


%module Quaternions

 // Quiet warnings about overloaded operators being ignored.
#pragma SWIG nowarn=362,389,401,509
%include <typemaps.i>
%include <stl.i>

%{
  #define SWIG_FILE_WITH_INIT
  #include <vector>
%}

// Use numpy below
%include <numpy.i>
%init %{
import_array();
%}
%pythoncode %{
  import numpy;
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
  #include <complex>
  #include "Quaternions.hpp"
  #include "IntegrateAngularVelocity.hpp"
%}


//////////////////////////////////////////////////////////////////////
//// The following translates between c++ and python types nicely ////
//////////////////////////////////////////////////////////////////////
%include "vector_typemaps.i"
namespace Quaternions {
  class Quaternion;
 };

// A missing precedence definition (a la `swig.swg`)
%define SWIG_TYPECHECK_QUATERNION         101     %enddef
%define SWIG_TYPECHECK_QUATERNION_ARRAY   1092    %enddef

// typecheck supports overloading
%typecheck(SWIG_TYPECHECK_QUATERNION) Quaternions::Quaternion {
  $1 = PySequence_Check($input) ? 1 : 0;
}
// Allow list input (with length 4)
%typemap(in) Quaternions::Quaternion
(Quaternions::Quaternion tmp, PyObject* item) {
  // A first sequence test
  if (!PySequence_Check($input)) {
    SWIG_exception(SWIG_TypeError, "expected a sequence for argument $argnum");
  }
  // Get sequence length
  Py_ssize_t pyseq_length = PySequence_Size($input);
  if (pyseq_length != 4) {
    SWIG_exception(SWIG_TypeError, "expected a sequence with length "	\
		   "4 for argument $argnum");
  }
  for (i = 0; i < pyseq_length; i++) {
    item = PySequence_GetItem($input, i);
    if(!SWIG_IsOK(SWIG_AsVal(double)(item, value))) {
      Py_DECREF(item);
      SWIG_exception(SWIG_TypeError, "expected items of sequence to be of type "\
		     "\"double\" in argument $argnum");
    }
    tmp[i] = value;
    Py_DECREF(item);
  }
  $1 = &tmp_vec;
}

// typecheck supports overloading
%typecheck(SWIG_TYPECHECK_QUATERNION) Quaternions::Quaternion& {
  if(!PySequence_Check($input) || !PySequence_Size($input)) {
    $1 = 0;
  } else {
    PyObject* item = PySequence_GetItem($input, 0);
    $1 = PyNumber_Check(item) ? 1 : 0;
  }
}
//Allow list input (with length 4)
%typemap(in) Quaternions::Quaternion&
(Quaternions::Quaternion tmp, PyObject* item, double value, Py_ssize_t i) {
  // A first sequence test
  if (!PySequence_Check($input)) {
    SWIG_exception(SWIG_TypeError, "expected a sequence for argument $argnum");
  }
  // Get sequence length
  Py_ssize_t pyseq_length = PySequence_Size($input);
  if (pyseq_length != 4) {
    SWIG_exception(SWIG_TypeError, "expected a sequence with length "	\
		   "4 for argument $argnum");
  }
  for (i = 0; i < pyseq_length; i++) {
    item = PySequence_GetItem($input, i);
    if(!SWIG_IsOK(SWIG_AsVal(double)(item, &value))) {
      Py_DECREF(item);
      SWIG_exception(SWIG_TypeError, "expected items of sequence to be of type "\
		     "\"double\" in argument $argnum");
    }
    tmp[i] = value;
    Py_DECREF(item);
  }
  $1 = &tmp;
}

// Return Quaternions as numpy arrays
%typemap(out) Quaternions::Quaternion {
  npy_intp dims[1] = { 4 };
  PyArrayObject* npy_arr = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  double* dat = (double*) PyArray_DATA(npy_arr);
  for (size_t i=0; i<4; ++i) { dat[i] = $1[i]; }
  $result = PyArray_Return(npy_arr);
}

// Return Quaternions as numpy arrays
%typemap(out) std::vector<Quaternions::Quaternion> {
  size_t result_size1 = $1.size();
  npy_intp dims[2] = {result_size1, 4};
  PyArrayObject* npy_arr = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  double* dat = (double*) PyArray_DATA(npy_arr);
  for (size_t i = 0; i < result_size1; ++i) {
    dat[i*4]   = (std::vector< Quaternions::Quaternion >($1))[i][0];
    dat[i*4+1] = (std::vector< Quaternions::Quaternion >($1))[i][1];
    dat[i*4+2] = (std::vector< Quaternions::Quaternion >($1))[i][2];
    dat[i*4+3] = (std::vector< Quaternions::Quaternion >($1))[i][3];
  }
  $result = PyArray_Return(npy_arr);
}


%typecheck(SWIG_TYPECHECK_QUATERNION_ARRAY) std::vector<Quaternions::Quaternion>& {
  // Check for nested sequence
  if(!PySequence_Check($input)) {
    // This is not a sequence at all
    $1 = 0;
  } else {
    if(!PySequence_Size($input)) {
      // The sequence has length 0...
      $1 = 1;
    } else {
      PyObject* item = PySequence_GetItem($input, 0);
      if(!PySequence_Check(item) || PySequence_Size(item)!=4) {
	// This is not a *nested* sequence, or the inner sequence
	// does not have length 4 (for a quaternion)
	$1 = 0;
      } else {
	// Check that what's inside this sequence is just a number
	PyObject* item2 = PySequence_GetItem(item, 0);
	$1 = PyNumber_Check(item2) ? 1 : 0;
      }
    }
  }
}
%fragment("SWIG_AsVal_frag(Quaternion)", "header") {
  SWIGINTERNINLINE int SWIG_AsVal(Quaternion)(SWIG_Object obj, Quaternions::Quaternion& val) {
    if(!PySequence_Check(obj) || PySequence_Size(obj)!=4) {
      return SWIG_TypeError;
    }
    double value;
    for(size_t i=0; i<4; ++i) {
      PyObject* component = PySequence_GetItem(obj, i);
      int res = SWIG_AsVal(double)(component, &value);
      if(!SWIG_IsOK(res)) {
	return res;
      }
      val[i] = value;
    }
    return 0;
  }
}
%typemap (in, fragment="SWIG_AsVal_frag(Quaternion)") std::vector<Quaternions::Quaternion>&
(std::vector<Quaternions::Quaternion> tmp_vec, PyObject* item, Quaternions::Quaternion value, Py_ssize_t i) {
  // Get sequence length
  Py_ssize_t pyseq_length = PySequence_Size($input);
  tmp_vec.reserve(pyseq_length);
  for (i=0; i<pyseq_length; i++) {
    item = PySequence_GetItem($input, i);
    if(!SWIG_IsOK(SWIG_AsVal(Quaternion)(item, value))) {
      Py_DECREF(item);
      SWIG_exception(SWIG_TypeError, "expected items of sequence to be of type " \
		     "\"Quaternion\" in argument $argnum");
    }
    tmp_vec.push_back(value);
    Py_DECREF(item);
  }
  $1 = &tmp_vec;
}



/////////////////////////////////////
//// Import the quaternion class ////
/////////////////////////////////////
// %ignore Quaternions::Quaternion::operator=;
%rename(__getitem__) Quaternions::Quaternion::operator [](const unsigned int) const;
%rename(__setitem__) Quaternions::Quaternion::operator [](const unsigned int);
%include "Quaternions.hpp"
%include "IntegrateAngularVelocity.hpp"
%extend Quaternions::Quaternion {
  // This function is called when printing a Quaternion object
  const char* __str__() {
    std::stringstream S;
    S << std::setprecision(14) << "["
      << $self->operator[](0) << ", "
      << $self->operator[](1) << ", "
      << $self->operator[](2) << ", "
      << $self->operator[](3) << "]";
    const std::string& tmp = S.str();
    const char* cstr = tmp.c_str();
    return cstr;
  }
  // This prints the Quaternion nicely at the prompt and allows nucer manipulations
  %pythoncode{
    def __repr__(self):
        return 'Quaternion('+repr(self[0])+', '+repr(self[1])+', '+repr(self[2])+', '+repr(self[3])+')'
    def __pow__(self, P) :
        return self.pow(P)
    __radd__ = __add__
    def __rsub__(self, t) :
        return -self+t
    __rmul__ = __mul__
    def __rdiv__(self, t) :
        return self.inverse()*t
  };
 };


/// Add utility functions that are specific to python.  Note that
/// these are defined in the Quaternions namespace.
%insert("python") %{


%}
