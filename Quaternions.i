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
  #include <iomanip>
  #include <complex>
  #include "Quaternions.hpp"
  #include "IntegrateAngularVelocity.hpp"
%}


//////////////////////////////////////////////////////////////////////
//// The following translates between c++ and python types nicely ////
//////////////////////////////////////////////////////////////////////
namespace Quaternions {
  class Quaternion;
};

// A missing precedence definition (a la `swig.swg`)
%define SWIG_TYPECHECK_QUATERNION         101     %enddef
%define SWIG_TYPECHECK_QUATERNION_ARRAY   1092    %enddef

// typecheck supports overloading
%typecheck(SWIG_TYPECHECK_QUATERNION) Quaternions::Quaternion {
  void* argp1 = 0;
  $1 = SWIG_IsOK(SWIG_ConvertPtr($input, &argp1, SWIGTYPE_p_Quaternions__Quaternion, 0 | 0 ));
  if(!$1) {
    // Accept it if this is a sequence of numbers with length 4
    if(PySequence_Check($input) && PySequence_Size($input)==4) {
      PyObject* item = PySequence_GetItem($input, 0);
      $1 = (PyFloat_Check(item) || PyInt_Check(item));
    }
  }
}
// Allow input as either a Quaternion or a sequence of length 4
%typemap(in) Quaternions::Quaternion
(Quaternions::Quaternion tmp, PyObject* item, Py_ssize_t i) {
  if(PySequence_Check($input)) {
    for(i=0; i<4; ++i) {
      item = PySequence_GetItem($input, i);
      if(!SWIG_IsOK(SWIG_AsVal(double)(item, &tmp[i]))) {
        Py_DECREF(item);
        SWIG_exception(SWIG_TypeError, "expected items of sequence to be of type "\
    		     "\"double\" in argument $argnum");
      }
      Py_DECREF(item);
    }
    $1 = &tmp;
  } else {
    void* ptmp;
    int res = SWIG_ConvertPtr($input, &ptmp, SWIGTYPE_p_Quaternions__Quaternion, 0 | 0);
    if (!SWIG_IsOK(res)) {
      SWIG_exception_fail(SWIG_ArgError(res), "expected argument $argnum to be of type \"Quaternion\".");
    }
    Quaternions::Quaternion* ptmp2 = reinterpret_cast< Quaternions::Quaternion * >(ptmp);
    tmp[0] = ptmp2->operator[](0);
    tmp[1] = ptmp2->operator[](1);
    tmp[2] = ptmp2->operator[](2);
    tmp[3] = ptmp2->operator[](3);
    $1 = &tmp;
  }
}

// typecheck supports overloading
%typecheck(SWIG_TYPECHECK_QUATERNION) const Quaternions::Quaternion& {
  void* argp1 = 0;
  $1 = SWIG_IsOK(SWIG_ConvertPtr($input, &argp1, SWIGTYPE_p_Quaternions__Quaternion, 0 | 0 ));
  if(!$1) {
    // Accept it if this is a sequence of numbers with length 4
    if(PySequence_Check($input) && PySequence_Size($input)==4) {
      PyObject* item = PySequence_GetItem($input, 0);
      $1 = (PyFloat_Check(item) || PyInt_Check(item));
    }
  }
}
// Allow input as either a Quaternion or a sequence of length 4
%typemap(in) const Quaternions::Quaternion&
(Quaternions::Quaternion tmp, PyObject* item, Py_ssize_t i) {
  if(PySequence_Check($input)) {
    for(i=0; i<4; ++i) {
      item = PySequence_GetItem($input, i);
      if(!SWIG_IsOK(SWIG_AsVal(double)(item, &tmp[i]))) {
        Py_DECREF(item);
        SWIG_exception(SWIG_TypeError, "expected items of sequence to be of type "\
    		     "\"double\" in argument $argnum");
      }
      Py_DECREF(item);
    }
    $1 = &tmp;
  } else {
    void* ptmp;
    int res = SWIG_ConvertPtr($input, &ptmp, SWIGTYPE_p_Quaternions__Quaternion, 0 | 0);
    if (!SWIG_IsOK(res)) {
      SWIG_exception_fail(SWIG_ArgError(res), "expected argument $argnum to be of type \"Quaternion\".");
    }
    Quaternions::Quaternion* ptmp2 = reinterpret_cast< Quaternions::Quaternion * >(ptmp);
    tmp[0] = ptmp2->operator[](0);
    tmp[1] = ptmp2->operator[](1);
    tmp[2] = ptmp2->operator[](2);
    tmp[3] = ptmp2->operator[](3);
    $1 = &tmp;
  }
}

// // Return vectors of Quaternions as numpy arrays
// %typemap(out) std::vector<Quaternions::Quaternion> {
//   size_t result_size = $1.size();
//   npy_intp dims[1] = {result_size};
//   PyArrayObject* npy_arr = (PyArrayObject*)PyArray_SimpleNewFromData(1, dims, NPY_OBJECT, (void*)&(*(&$1))[0]);
//   Py_INCREF($1);
//   // Quaternions::Quaternion* dat = (Quaternions::Quaternion*) PyArray_DATA(npy_arr);
//   // for (size_t i = 0; i < result_size1; ++i) {
//   //   dat[i]   = Quaternions::Quaternion($1)[i];
//   // }
//   $result = PyArray_Return(npy_arr);
// }

%typecheck(SWIG_TYPECHECK_QUATERNION_ARRAY) std::vector<Quaternions::Quaternion>& {
  // Check for sequence
  if(!PySequence_Check($input)) {
    // This is not a sequence at all
    $1 = 0;
  } else {
    if(!PySequence_Size($input)) {
      // The sequence has length 0...
      $1 = 1;
    } else {
      // Check that the first element is a quaternion
      PyObject* item = PySequence_GetItem($input, 0);
      void* argp1 = 0;
      $1 = SWIG_IsOK(SWIG_ConvertPtr(item, &argp1, SWIGTYPE_p_Quaternions__Quaternion, 0 | 0 ));
    }
  }
}
%typemap (in) std::vector<Quaternions::Quaternion>&
(std::vector<Quaternions::Quaternion> tmp_vec, Quaternions::Quaternion tmp, void* ptmp, PyObject* item, Py_ssize_t i) {
  // Get sequence length
  Py_ssize_t pyseq_length = PySequence_Size($input);
  tmp_vec.reserve(pyseq_length);
  for (i=0; i<pyseq_length; i++) {
    item = PySequence_GetItem($input, i);
    int res = SWIG_ConvertPtr(item, &ptmp, SWIGTYPE_p_Quaternions__Quaternion, 0 | 0);
    if (!SWIG_IsOK(res)) {
      Py_DECREF(item);
      SWIG_exception_fail(SWIG_ArgError(res), "array element failed to convert to \"Quaternion\".");
    }
    tmp = Quaternions::Quaternion(*(reinterpret_cast< Quaternions::Quaternion * >(ptmp)));
    Py_DECREF(item);
    tmp_vec.push_back(tmp);
  }
  $1 = &tmp_vec;
}



/////////////////////////////////////
//// Import the quaternion class ////
/////////////////////////////////////
// %ignore Quaternions::Quaternion::operator=;
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
    S << std::setprecision(14) << "["
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
    S << std::setprecision(16) << "Quaternion("
      << $self->operator[](0) << ", "
      << $self->operator[](1) << ", "
      << $self->operator[](2) << ", "
      << $self->operator[](3) << ")";
    const std::string& tmp = S.str();
    const char* cstr = tmp.c_str();
    return cstr;
  }

  // %pythoncode{
  //   def __repr__(self):
  //       return 'Quaternion('+repr(self[0])+', '+repr(self[1])+', '+repr(self[2])+', '+repr(self[3])+')'
  //   def __pow__(self, P) :
  //       return self.pow(P)
  //   __radd__ = __add__
  //   def __rsub__(self, t) :
  //       return -self+t
  //   __rmul__ = __mul__
  //   def __rdiv__(self, t) :
  //       return self.inverse()*t
  // };
 };

// The following line is needed to ensure that SWIG knows how to
// destruct pointers to vectors of Quaternions
%template(_QuaternionVec) std::vector<Quaternions::Quaternion>;


%include "vector_typemaps.i"


/// Add utility functions that are specific to python.  Note that
/// these are defined in the Quaternions namespace.
%insert("python") %{


%}
