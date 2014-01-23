// A missing precedence definition (a la `swig.swg`)
%define SWIG_TYPECHECK_COMPLEX_ARRAY      1091    %enddef

// Make sure std::complex numbers are dealt with appropriately
%include <std_complex.i>
%{
typedef std::complex<double> std_complex_double;
%}
typedef std::complex<double> std_complex_double;
%{
#define SWIG_AsVal_std_complex_double SWIG_AsVal_std_complex_Sl_double_Sg_
%}

%define IN_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(TYPE, TYPE_UPPER, ARG_NAME, \
					    NUMPY_TYPE, TYPE_NAME, DESCR)
%typecheck(SWIG_TYPECHECK_ ## TYPE_UPPER ## _ARRAY)  \
const std::vector<TYPE>&  ARG_NAME {
  $1 = false;
  if(PyArray_Check($input)) {
    $1 = (PyArray_NDIM(reinterpret_cast<const PyArrayObject*>($input))==1);
  } else if(PyList_Check($input)) {
    if(PyList_Size($input)==0) {
      $1 = true;
    } else {
      PyObject* item = PySequence_GetItem($input, 0);
      TYPE* temp;
      $1 = SWIG_IsOK(SWIG_AsVal(TYPE)(item, temp));
    }
  }
}
%typemap(in) const std::vector<TYPE>& ARG_NAME (std::vector<TYPE> temp) {
  if(PyArray_Check($input)) {
    PyArrayObject *xa = reinterpret_cast<PyArrayObject*>($input);
    if(PyArray_TYPE(xa) != NUMPY_TYPE) {
      SWIG_exception(SWIG_TypeError, "(1) numpy array of 'TYPE_NAME' expected."	\
		     " Make sure that the numpy array use dtype=DESCR.");
    }
    const std::size_t size = PyArray_DIM(xa, 0);
    temp.resize(size);
    TYPE* array = static_cast<TYPE*>(PyArray_DATA(xa));
    if(PyArray_ISCONTIGUOUS(xa)) {
      std::copy(array, array + size, temp.begin());
    } else {
      const npy_intp strides = PyArray_STRIDE(xa, 0)/sizeof(TYPE);
      for (std::size_t i = 0; i < size; i++)
	temp[i] = array[i*strides];
    }
  } else if(PySequence_Check($input)) {
    Py_ssize_t size = PySequence_Size($input);
    temp.resize(size);
    PyObject* item;
    for(Py_ssize_t i=0; i<size; ++i) {
      item = PySequence_GetItem($input, i);
      if(!SWIG_IsOK(SWIG_AsVal(TYPE)(item, &temp[i]))) {
        Py_DECREF(item);
        SWIG_exception(SWIG_TypeError, "expected items of sequence to be of type "\
		       "\"TYPE\" in argument $argnum");
      }
      Py_DECREF(item);
    }
  } else {
    SWIG_exception(SWIG_TypeError, "(2) numpy array of 'TYPE_NAME' expected. " \
		   "Make sure that the numpy array use dtype=DESCR.");
  }
  $1 = &temp;
}
%enddef

%define IN_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_PRIMITIVES(TYPE, TYPE_UPPER, ARG_NAME, \
							  NUMPY_TYPE, TYPE_NAME, DESCR)
%typecheck(SWIG_TYPECHECK_ ## TYPE_UPPER ## _ARRAY)  \
const std::vector<std::vector<TYPE> >&  ARG_NAME {
  $1 = false;
  if(PyArray_Check($input)) {
    $1 = (PyArray_NDIM(reinterpret_cast<const PyArrayObject*>($input))==2);
  } else if(PyList_Check($input)) {
    if(PyList_Size($input)==0) {
      $1 = true;
    } else {
      PyObject* item0 = PySequence_GetItem($input, 0);
      if(PyList_Size(item0)==0) {
	$1 = true;
      } else {
	PyObject* item1 = PySequence_GetItem(item0, 0);
	TYPE* temp;
	$1 = SWIG_IsOK(SWIG_AsVal(TYPE)(item1, temp));
      }
    }
  }
}
%typemap(in) const std::vector<std::vector<TYPE> >& ARG_NAME (std::vector<std::vector<TYPE> > temp) {
  if(PyArray_Check($input)) {
    PyArrayObject *xa = reinterpret_cast<PyArrayObject*>($input);
    if(PyArray_TYPE(xa) != NUMPY_TYPE) {
      SWIG_exception(SWIG_TypeError, "(2) numpy array of 'TYPE_NAME' expected."	\
		     " Make sure that the numpy array use dtype=DESCR.");
    }
    const std::size_t size0 = PyArray_DIM(xa, 0);
    const std::size_t size1 = PyArray_DIM(xa, 1);
    temp.resize(size0);
    for(unsigned int i=0; i<size0; ++i) {
      temp[i].resize(size1);
    }
    TYPE* array = static_cast<TYPE*>(PyArray_DATA(xa));
    const npy_intp strides0 = PyArray_STRIDE(xa, 0)/sizeof(TYPE);
    const npy_intp strides1 = PyArray_STRIDE(xa, 1)/sizeof(TYPE);
    for (std::size_t i = 0; i < size0; ++i) {
      for (std::size_t j = 0; j< size1; ++j) {
	temp[i][j] = array[i*strides0+j*strides1];
      }
    }
  } else if(PySequence_Check($input)) {
    Py_ssize_t size0 = PySequence_Size($input);
    temp.resize(size0);
    PyObject* item0;
    PyObject* item1;
    for(Py_ssize_t i=0; i<size0; ++i) {
      item0 = PySequence_GetItem($input, i);
      Py_ssize_t size1 = PySequence_Size(item0);
      temp[i].resize(size1);
      for(Py_ssize_t j=0; j<size1; ++j) {
	item1 = PySequence_GetItem(item0, j);
	if(!SWIG_IsOK(SWIG_AsVal(TYPE)(item1, &temp[i][j]))) {
	  Py_DECREF(item1);
	  SWIG_exception(SWIG_TypeError, "expected items of sequence to be sequences of type " \
			 "\"TYPE\" in argument $argnum");
	}
	Py_DECREF(item1);
      }
    }
  } else {
    SWIG_exception(SWIG_TypeError, "(1) numpy array of 'TYPE_NAME' expected.");
  }
  $1 = &temp;
}
%enddef


%define ARGOUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(TYPE, TYPE_UPPER, ARG_NAME, \
						NUMPY_TYPE)
%typemap (in,numinputs=0) std::vector<TYPE>& ARG_NAME (std::vector<TYPE> vec_temp)
{
  $1 = &vec_temp;
}
%typemap(argout) std::vector<TYPE>& ARG_NAME
{
  npy_intp size = $1->size();
  PyArrayObject *ret = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size, NUMPY_TYPE));
  TYPE* data = static_cast<TYPE*>(PyArray_DATA(ret));
  for (npy_intp i = 0; i < size; ++i)
    data[i] = (*$1)[i];
  // Append the output to $result
  %append_output(PyArray_Return(ret));
}
%enddef

%define ARGOUT_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_PRIMITIVES(TYPE, TYPE_UPPER, ARG_NAME, \
						NUMPY_TYPE)
%typemap (in,numinputs=0) std::vector<std::vector<TYPE> >& ARG_NAME (std::vector<std::vector<TYPE> > vec_temp)
{
  $1 = &vec_temp;
}
%typemap(argout) std::vector<std::vector<TYPE> >& ARG_NAME
{
  npy_intp result_size = $1->size();
  npy_intp result_size2 = (result_size>0 ? (*$1)[0].size() : 0);
  npy_intp dims[2] = { result_size, result_size2 };
  PyArrayObject* npy_arr = (PyArrayObject*)PyArray_SimpleNew(2, dims, NUMPY_TYPE);
  TYPE* dat = static_cast<TYPE*>(PyArray_DATA(npy_arr));
  for (npy_intp i = 0; i < result_size; ++i) { for (npy_intp j = 0; j < result_size2; ++j) { dat[i*result_size2+j] = (*$1)[i][j]; } }
  %append_output(PyArray_Return(npy_arr));
}
%enddef


// Some fragments used in the array typemaps below
%fragment("PyInteger_Check", "header")
{
  SWIGINTERNINLINE bool PyInteger_Check(PyObject* in)
  {
    return  PyInt_Check(in) || (PyArray_CheckScalar(in) &&
				PyArray_IsScalar(in,Integer));
  }
}

//-----------------------------------------------------------------------------
// Home brewed versions of the SWIG provided SWIG_AsVal(Type). These are needed
// as long as we need the PyInteger_Check. Whenever Python 2.6 is not supported
// we can scrap them.
//-----------------------------------------------------------------------------
#define Py_convert_frag(Type) "Py_convert_" {Type}
%fragment("Py_convert_std_size_t", "header", fragment="PyInteger_Check")
{
  // A check for integer
  SWIGINTERNINLINE bool Py_convert_std_size_t(PyObject* in, std::size_t& value)
  {
    if (!(PyInteger_Check(in) && PyInt_AS_LONG(in)>=0))
      return false;
    value = static_cast<std::size_t>(PyInt_AS_LONG(in));
    return true;
  }
}
%fragment("Py_convert_double", "header") {
  // A check for float and converter for double
  SWIGINTERNINLINE bool Py_convert_double(PyObject* in, double& value)
  {
    return SWIG_AsVal(double)(in, &value);
  }
}
%fragment("Py_convert_int", "header", fragment="PyInteger_Check") {
  // A check for int and converter for int
  SWIGINTERNINLINE bool Py_convert_int(PyObject* in, int& value)
  {
    if (!PyInteger_Check(in))
      return false;
    value = static_cast<int>(PyInt_AS_LONG(in));
    return true;
  }
}
%fragment("Py_convert_uint", "header", fragment="PyInteger_Check") {
  // A check for int and converter to uint
  SWIGINTERNINLINE bool Py_convert_uint(PyObject* in, unsigned int& value)
  {
    if (!(PyInteger_Check(in) && PyInt_AS_LONG(in)>=0))
      return false;
    value = static_cast<unsigned int>(PyInt_AS_LONG(in));
    return true;
  }
}

%define PY_SEQUENCE_OF_SCALARS_TO_VECTOR_OF_PRIMITIVES(TYPE, TYPE_UPPER, \
					      ARG_NAME, TYPE_NAME, SEQ_LENGTH)
%typecheck(SWIG_TYPECHECK_ ## TYPE_UPPER ## _ARRAY) std::vector<TYPE> ARG_NAME
{
  $1 = PySequence_Check($input) ? 1 : 0;
}
%typemap (in, fragment=Py_convert_frag(TYPE_NAME)) std::vector<TYPE> ARG_NAME
(std::vector<TYPE> tmp_vec, PyObject* item, TYPE value, std::size_t i)
{
  // A first sequence test
  if (!PySequence_Check($input))
  {
    SWIG_exception(SWIG_TypeError, "expected a sequence for argument $argnum");
  }
  // Get sequence length
  Py_ssize_t pyseq_length = PySequence_Size($input);
  if (SEQ_LENGTH >= 0 && pyseq_length > SEQ_LENGTH)
  {
    SWIG_exception(SWIG_TypeError, "expected a sequence with length "	\
		   "SEQ_LENGTH for argument $argnum");
  }
  tmp_vec.reserve(pyseq_length);
  for (i = 0; i < pyseq_length; i++)
  {
    item = PySequence_GetItem($input, i);
    if(!SWIG_IsOK(Py_convert_ ## TYPE_NAME(item, value)))
    {
      Py_DECREF(item);
      SWIG_exception(SWIG_TypeError, "expected items of sequence to be of type "\
		     "\"TYPE_NAME\" in argument $argnum");
    }
    tmp_vec.push_back(value);
    Py_DECREF(item);
  }
  $1 = tmp_vec;
}
%enddef

%define OUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(TYPE, NUMPY_TYPE)
%typemap(out) std::vector<TYPE>
{
  // RANDOMSTRING1
  npy_intp adims = $1.size();
  $result = PyArray_SimpleNew(1, &adims, NUMPY_TYPE);
  TYPE* data = static_cast<TYPE*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>($result)));
  std::copy($1.begin(), $1.end(), data);
}
%enddef

%define OUT_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_PRIMITIVES(TYPE, NUMPY_TYPE)
%typemap(out) std::vector<std::vector<TYPE> >
{
  // RANDOMSTRING1
  npy_intp adim1 = (&$1)->size();
  npy_intp adim2 = (adim1>0 ? (*(&$1))[0].size() : 0);
  npy_intp adims[2] = {adim1, adim2};
  $result = PyArray_SimpleNew(2, adims, NUMPY_TYPE);
  TYPE* data = static_cast<TYPE*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>($result)));
  for(npy_intp i=0; i<adim1; ++i) {
    for(npy_intp j=0; j<adim2; ++j) {
      data[i*adim2+j] = (*(&$1))[i][j];
    }
  }
}
%enddef

%define OUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES_REF(TYPE, NUMPY_TYPE)
%typemap(out) std::vector<TYPE>&
{
  // RANDOMSTRING2
  npy_intp adims = $1->size();
  $result = PyArray_SimpleNew(1, &adims, NUMPY_TYPE);
  TYPE* data = static_cast<TYPE*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>($result)));
  std::copy($1->begin(), $1->end(), data);
}
%enddef

%define OUT_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_PRIMITIVES_REF(TYPE, NUMPY_TYPE)
%typemap(out) std::vector<std::vector<TYPE> >&
{
  // RANDOMSTRING2
  npy_intp adim1 = $1->size();
  npy_intp adim2 = (adim1>0 ? (*$1)[0].size() : 0);
  npy_intp adims[2] = {adim1, adim2};
  $result = PyArray_SimpleNew(2, adims, NUMPY_TYPE);
  TYPE* data = static_cast<TYPE*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>($result)));
  for(npy_intp i=0; i<adim1; ++i) {
    for(npy_intp j=0; j<adim2; ++j) {
      data[i*adim2+j] = (*$1)[i][j];
    }
  }
}
%enddef

%define OUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES_CONST_REF(TYPE, NUMPY_TYPE)
%typemap(out) std::vector<TYPE> const&
{
  // RANDOMSTRING3
  npy_intp adims = $1->size();
  $result = PyArray_SimpleNew(1, &adims, NUMPY_TYPE);
  TYPE* data = static_cast<TYPE*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>($result)));
  std::copy($1->begin(), $1->end(), data);
}
%enddef

%define OUT_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_PRIMITIVES_CONST_REF(TYPE, NUMPY_TYPE)
%typemap(out) std::vector<std::vector<TYPE> > const&
{
  // RANDOMSTRING3
  npy_intp adim1 = $1->size();
  npy_intp adim2 = (adim1>0 ? (*$1)[0].size() : 0);
  npy_intp adims[2] = {adim1, adim2};
  $result = PyArray_SimpleNew(2, adims, NUMPY_TYPE);
  TYPE* data = static_cast<TYPE*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>($result)));
  for(npy_intp i=0; i<adim1; ++i) {
    for(npy_intp j=0; j<adim2; ++j) {
      data[i*adim2+j] = (*$1)[i][j];
    }
  }
}
%enddef


IN_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(int, INT32, , NPY_INT, int, intc)
IN_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(double, DOUBLE, , NPY_DOUBLE, double, double)
IN_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(std_complex_double, COMPLEX, , NPY_CDOUBLE, complex, complex)

IN_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_PRIMITIVES(int, INT32, , NPY_INT, int, intc)
IN_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_PRIMITIVES(double, DOUBLE, , NPY_DOUBLE, double, double)
IN_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_PRIMITIVES(std_complex_double, COMPLEX, , NPY_CDOUBLE, complex, complex)

ARGOUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(int, INT32, ARGOUT, NPY_INT)
ARGOUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(double, DOUBLE, ARGOUT, NPY_DOUBLE)
ARGOUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(std_complex_double, COMPLEX, ARGOUT, NPY_CDOUBLE)

ARGOUT_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_PRIMITIVES(int, INT32, ARGOUT, NPY_INT)
ARGOUT_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_PRIMITIVES(double, DOUBLE, ARGOUT, NPY_DOUBLE)
ARGOUT_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_PRIMITIVES(std_complex_double, COMPLEX, ARGOUT, NPY_CDOUBLE)

PY_SEQUENCE_OF_SCALARS_TO_VECTOR_OF_PRIMITIVES(int, INT32, IN_PY_SEQUENCE, int, -1)
PY_SEQUENCE_OF_SCALARS_TO_VECTOR_OF_PRIMITIVES(double, DOUBLE, IN_PY_SEQUENCE, double, -1)
PY_SEQUENCE_OF_SCALARS_TO_VECTOR_OF_PRIMITIVES(std_complex_double, COMPLEX, IN_PY_SEQUENCE, std_complex_double, -1)

OUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(int, NPY_INT)
OUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(double, NPY_DOUBLE)
OUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(std_complex_double, NPY_CDOUBLE)

OUT_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_PRIMITIVES(int, NPY_INT)
OUT_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_PRIMITIVES(double, NPY_DOUBLE)
OUT_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_PRIMITIVES(std_complex_double, NPY_CDOUBLE)

OUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES_REF(int, NPY_INT)
OUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES_REF(double, NPY_DOUBLE)
OUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES_REF(std_complex_double, NPY_CDOUBLE)

OUT_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_PRIMITIVES_REF(int, NPY_INT)
OUT_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_PRIMITIVES_REF(double, NPY_DOUBLE)
OUT_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_PRIMITIVES_REF(std_complex_double, NPY_CDOUBLE)

OUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES_CONST_REF(int, NPY_INT)
OUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES_CONST_REF(double, NPY_DOUBLE)
OUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES_CONST_REF(std_complex_double, NPY_CDOUBLE)

OUT_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_PRIMITIVES_CONST_REF(int, NPY_INT)
OUT_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_PRIMITIVES_CONST_REF(double, NPY_DOUBLE)
OUT_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_PRIMITIVES_CONST_REF(std_complex_double, NPY_CDOUBLE)
