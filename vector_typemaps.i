// A missing precedence definition (a la `swig.swg`)
%define SWIG_TYPECHECK_COMPLEX_ARRAY      1091    %enddef


// Make sure std::complex numbers are dealt with appropriately
%include <std_complex.i>
typedef std::complex<double> std_complex_double;


%define IN_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(TYPE, TYPE_UPPER, ARG_NAME, \
					    NUMPY_TYPE, TYPE_NAME, DESCR)
%typecheck(SWIG_TYPECHECK_ ## TYPE_UPPER ## _ARRAY)  \
const std::vector<TYPE>&  ARG_NAME
{
  $1 = PyArray_Check($input) ? 1 : 0;
}
%typemap(in) const std::vector<TYPE>& ARG_NAME (std::vector<TYPE> temp)
{
  {
    if (!PyArray_Check($input))
    {
      SWIG_exception(SWIG_TypeError, "(2) numpy array of 'TYPE_NAME' expected. "\
		     "Make sure that the numpy array use dtype=DESCR.");
    }
    PyArrayObject *xa = reinterpret_cast<PyArrayObject*>($input);
    if ( PyArray_TYPE(xa) != NUMPY_TYPE )
    {
      SWIG_exception(SWIG_TypeError, "(1) numpy array of 'TYPE_NAME' expected."	\
		     " Make sure that the numpy array use dtype=DESCR.");
    }
    const std::size_t size = PyArray_DIM(xa, 0);
    temp.resize(size);
    TYPE* array = static_cast<TYPE*>(PyArray_DATA(xa));
    if (PyArray_ISCONTIGUOUS(xa))
    {
      std::copy(array, array + size, temp.begin());
    }
    else
    {
      const npy_intp strides = PyArray_STRIDE(xa, 0)/sizeof(TYPE);
      for (std::size_t i = 0; i < size; i++)
	temp[i] = array[i*strides];
    }
    $1 = &temp;
  }
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
  for (int i = 0; i < size; ++i)
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
  for (size_t i = 0; i < result_size; ++i) { for (size_t j = 0; j < result_size2; ++j) { dat[i*result_size2+j] = (*$1)[i][j]; } }
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
  npy_intp adims = $1.size();
  $result = PyArray_SimpleNew(1, &adims, NUMPY_TYPE);
  TYPE* data = static_cast<TYPE*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>($result)));
  std::copy($1.begin(), $1.end(), data);
}
%enddef


ARGOUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(int, INT32, ARGOUT, NPY_INT)
ARGOUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(double, DOUBLE, ARGOUT, NPY_DOUBLE)
ARGOUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(std_complex_double, COMPLEX, ARGOUT, NPY_CDOUBLE)
/* ARGOUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(Quaternions::Quaternion, QUATERNION, ARGOUT, NPY_OBJECT) */

ARGOUT_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_PRIMITIVES(int, INT32, ARGOUT, NPY_INT)
ARGOUT_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_PRIMITIVES(double, DOUBLE, ARGOUT, NPY_DOUBLE)
ARGOUT_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_PRIMITIVES(std_complex_double , COMPLEX, ARGOUT, NPY_CDOUBLE)
/* ARGOUT_TYPEMAP_STD_VECTOR_OF_STD_VECTOR_OF_PRIMITIVES(Quaternions::Quaternion, QUATERNION, ARGOUT, NPY_OBJECT) */

IN_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(int, INT32, IN, NPY_INT, int, intc)
IN_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(double, DOUBLE, IN, NPY_DOUBLE, double, double)
IN_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(std_complex_double , COMPLEX, IN, NPY_CDOUBLE, complex, complex)
/* IN_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(Quaternions::Quaternion, QUATERNION, IN, NPY_OBJECT, Quaternion, Quaternion) */

PY_SEQUENCE_OF_SCALARS_TO_VECTOR_OF_PRIMITIVES(int, INT32, IN_PY_SEQUENCE, int, -1)
PY_SEQUENCE_OF_SCALARS_TO_VECTOR_OF_PRIMITIVES(double, DOUBLE, IN_PY_SEQUENCE, double, -1)
PY_SEQUENCE_OF_SCALARS_TO_VECTOR_OF_PRIMITIVES(std_complex_double, COMPLEX, IN_PY_SEQUENCE, std_complex_double, -1)
/* PY_SEQUENCE_OF_SCALARS_TO_VECTOR_OF_PRIMITIVES(Quaternions::Quaternion, QUATERNION, IN_PY_SEQUENCE, Quaternions::Quaternion, -1) */

OUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(int, NPY_INT)
OUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(double, NPY_DOUBLE)
OUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(std_complex_double, NPY_CDOUBLE)
/* OUT_TYPEMAP_STD_VECTOR_OF_PRIMITIVES(Quaternions::Quaternion, NPY_OBJECT) */
