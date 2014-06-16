// -*- c++ -*-

//////////////////////////////////////////////////////////////////////
//// The following translates between c++ and python types nicely ////
//////////////////////////////////////////////////////////////////////

#ifndef SWIGIMPORTED
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
#endif



// A missing precedence definition (a la `swig.swg`)
%define SWIG_TYPECHECK_QUATERNION         101     %enddef
%define SWIG_TYPECHECK_QUATERNION_ARRAY   1092    %enddef



// typecheck supports overloading
%typecheck(SWIG_TYPECHECK_QUATERNION) Quaternions::Quaternion {
  void* argp1 = 0;
  // If this is a SWIG-wrapped Quaternion, accept it
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
  // If this is a SWIG-wrapped Quaternion, accept it
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

%typemap (in,numinputs=0) Quaternions::Quaternion& Quaternion_argout (Quaternions::Quaternion quat_temp) {
  $1 = &quat_temp;
}
%typemap(argout) Quaternions::Quaternion& Quaternion_argout {
  PyObject* qobj = SWIG_NewPointerObj((new Quaternions::Quaternion(*$1)),
                                      SWIGTYPE_p_Quaternions__Quaternion, SWIG_POINTER_OWN);
  if(!qobj) {SWIG_fail;}
  Py_INCREF(qobj);
  %append_output(qobj);
}


// typecheck supports overloading
%typecheck(SWIG_TYPECHECK_QUATERNION_ARRAY) const std::vector<Quaternions::Quaternion>& {
  $1 = false;
  if(PySequence_Check($input)) { // If this is a python sequence...
    if(PySequence_Size($input)==0) { // and has zero length, accept
      $1 = true;
    } else { // otherwise...
      PyObject* item0 = PySequence_GetItem($input, 0);
      if(PySequence_Check(item0) && PySequence_Size(item0)==4) { // Accept if it's a sequence of sequences with length 4
        PyObject* item00 = PySequence_GetItem(item0, 0);
        $1 = (PyFloat_Check(item00) || PyInt_Check(item00));
      } else { // Accept if it's a sequence of SWIG-wrapped Quaternions
        void* p = 0;
        $1 = SWIG_IsOK(SWIG_ConvertPtr(item0, &p, SWIGTYPE_p_Quaternions__Quaternion, 0 | 0 ));
      }
    }
  // } else {
  //   // Accept if it's a SWIG-wrapped vector<Quaternion>
  //   void* p = 0;
  //   $1 = SWIG_IsOK(SWIG_ConvertPtr($input, &p, SWIGTYPE_p_std__vectorT_Quaternions__Quaternion_std__allocatorT_Quaternions__Quaternion_t_t, 0 | 0 ));
  }
}
// Allow input as either a Quaternion or a sequence of length 4
%typemap(in) const std::vector<Quaternions::Quaternion>&
(std::vector<Quaternions::Quaternion> tmp, PyObject* itemi, PyObject* itemij, void* p, Py_ssize_t size, Py_ssize_t i, Py_ssize_t j,  int res) {
  if(PySequence_Check($input)) { // If this is a python sequence...
    if(PySequence_Size($input)==0) { // and has zero length...
      tmp = std::vector<Quaternions::Quaternion>(0);
    } else { // otherwise...
      size = PySequence_Size($input);
      tmp = std::vector<Quaternions::Quaternion>(size);
      for(i=0; i<size; ++i) {
        itemi = PySequence_GetItem($input, i);
        if(PySequence_Check(itemi) && PySequence_Size(itemi)==4) { // Accept if it's a sequence of sequences with length 4
          for(j=0; j<4; ++j) {
            itemij = PySequence_GetItem(itemi, j);
            SWIG_AsVal(double)(itemij, &(tmp[i][j]));
          }
        } else { // Accept if it's a sequence of SWIG-wrapped Quaternions
          p = 0;
          res = SWIG_IsOK(SWIG_ConvertPtr(itemi, &p, SWIGTYPE_p_Quaternions__Quaternion, 0 | 0 ));
          if(!res) {
            SWIG_exception_fail(SWIG_ArgError(res), "expected argument $argnum to be a sequence of objects of type \"Quaternion\".");
          }
          tmp[i] = *((Quaternions::Quaternion*)p);
        }
      }
    }
  // } else {
  //   // Accept if it's a SWIG-wrapped vector<Quaternion>
  //   void* p = &tmp;
  //   res = SWIG_IsOK(SWIG_ConvertPtr($input, &p, SWIGTYPE_p_std__vectorT_Quaternions__Quaternion_std__allocatorT_Quaternions__Quaternion_t_t, 0 | 0 ));
  //   if(!res) {
  //     SWIG_exception_fail(SWIG_ArgError(res), "expected argument $argnum to be of type \"vector<Quaternion>\".");
  //   }
  }
  $1 = &tmp;
}

%typemap(out) std::vector<Quaternions::Quaternion> {
  npy_intp size = $1.size();
  $result = PyArray_SimpleNew(1, &size, NPY_OBJECT);
  PyObject** data = static_cast<PyObject**>(PyArray_DATA((PyArrayObject*)$result));
  for(npy_intp i=0; i<size; ++i) {
    PyObject* qobj = SWIG_NewPointerObj((new Quaternions::Quaternion((*(&$1))[i])),
                                        SWIGTYPE_p_Quaternions__Quaternion, SWIG_POINTER_OWN);
    if(!qobj) {SWIG_fail;}
    Py_INCREF(qobj);
    data[i] = qobj;
  }
}

%typemap(out) std::vector<Quaternions::Quaternion>& {
  npy_intp size = $1->size();
  $result = PyArray_SimpleNew(1, &size, NPY_OBJECT);
  PyObject** data = static_cast<PyObject**>(PyArray_DATA((PyArrayObject*)$result));
  for(npy_intp i=0; i<size; ++i) {
    PyObject* qobj = SWIG_NewPointerObj((new Quaternions::Quaternion((*$1)[i])),
                                        SWIGTYPE_p_Quaternions__Quaternion, SWIG_POINTER_OWN);
    if(!qobj) {SWIG_fail;}
    Py_INCREF(qobj);
    data[i] = qobj;
  }
}

%typemap (in,numinputs=0) std::vector<Quaternions::Quaternion>& Quaternions_argout (std::vector<Quaternions::Quaternion> vec_temp) {
  $1 = &vec_temp;
}
%typemap(argout) std::vector<Quaternions::Quaternion>& Quaternions_argout {
  npy_intp size = $1->size();
  PyArrayObject *npy_arr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size, NPY_OBJECT));
  PyObject** data = static_cast<PyObject**>(PyArray_DATA(npy_arr));
  for(npy_intp i=0; i<size; ++i) {
    PyObject* qobj = SWIG_NewPointerObj((new Quaternions::Quaternion((*$1)[i])),
                                        SWIGTYPE_p_Quaternions__Quaternion, SWIG_POINTER_OWN);
    if(!qobj) {SWIG_fail;}
    Py_INCREF(qobj);
    data[i] = qobj;
  }
  %append_output(PyArray_Return(npy_arr));
}



#ifndef SWIGIMPORTED

%include <std_vector.i>
%include "vector_typemaps.i"
namespace std {
  // %template(vectorq) vector<Quaternions::Quaternion>;
  // %template(vectord) vector<double>;
  // %template(vectorvectord) vector<vector<double> >;
};

#endif
