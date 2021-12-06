#ifndef NPMD_UTILS_H
#define NPMD_UTILS_H

#include<Python.h>
#include<numpy/arrayobject.h>
#include<numpy/ndarraytypes.h>
#include<stdlib.h>
#include<stdbool.h>



PyArrayObject* unwrap_array(PyObject* obj, int typenum, int expected_ndims);

npy_float unwrap_float(PyObject* obj);

npy_double unwrap_double(PyObject* obj);

npy_int unwrap_int(PyObject* obj, int* out);

npy_longlong unwrap_longlong(PyObject* obj);

npy_bool unwrap_bool(PyObject* obj);


#endif

