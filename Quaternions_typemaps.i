// -*- c++ -*-

//////////////////////////////////////////////////////////////////////
//// The following translates between c++ and python types nicely ////
//////////////////////////////////////////////////////////////////////

%include <typemaps.i>
%include <stl.i>

// Use numpy below
%include <numpy.i>
%init %{
  import_array();
%}
%pythoncode %{
  import numpy;
%}

namespace Quaternions {
  class Quaternion;
};
namespace std {
  %template(vectorq) vector<Quaternions::Quaternion>;
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
    tmp[0] = (*ptmp2)[0];
    tmp[1] = (*ptmp2)[1];
    tmp[2] = (*ptmp2)[2];
    tmp[3] = (*ptmp2)[3];
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
    tmp[0] = (*ptmp2)[0];
    tmp[1] = (*ptmp2)[1];
    tmp[2] = (*ptmp2)[2];
    tmp[3] = (*ptmp2)[3];
    $1 = &tmp;
  }
}

%include <std_vector.i>
%include "vector_typemaps.i"
