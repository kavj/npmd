#include<Python.h>
#include<numpy/arrayobject.h>
#include<numpy/ndarraytypes.h>
#include<stdlib.h>
#include<stdbool.h>
#include "npmd_utils.h"


PyArrayObject* unwrap_array(PyObject* obj, int typenum, int expected_ndims) {
    if (obj == NULL) {
        PyErr_SetString(PyExc_TypeError, "Unknown error.\n");
        return NULL;
    }

    PyArrayObject* out = (PyArrayObject*) PyArray_FROM_OTF(obj, typenum, NPY_ARRAY_CARRAY_RO);

    if (out == NULL) {
        // error should be already set
        return NULL;
    }

    int actual_ndims = PyArray_NDIM(out);

    if (actual_ndims != expected_ndims) {
        char buffer[128];
        sprintf(buffer, "Dimension mismatch. Expected array with %d dimensions, received %d dimensions", expected_ndims, actual_ndims);
        PyErr_SetString(PyExc_ValueError, buffer);
        return NULL;
    }

    return out;
}


npy_float unwrap_float(PyObject* obj) {
    if (obj == NULL) {
        return -1;
    }
    
    if (PyArray_IsPythonNumber(obj)) {
        // disambiguate type
        if (PyFloat_CheckExact(obj)) {
            npy_float out = (npy_float) PyFloat_AsDouble(obj);
            return out;
        }

        else if (PyLong_Check(obj)) {
            npy_float out = (npy_float) PyLong_AsDouble(obj);
            return out;
        }

        else {  // don't coerce any other builtin type
            PyErr_SetString(PyExc_TypeError, "Unsupported numerical python type for expected type float or np.float32.\n")
            return -1;
        }
    }

    else {
        // Check for exact numpy type
        if (PyArray_CheckScalar(obj)) {
            PyArray_Descr* outcode = PyArray_DescrFromType(NPY_FLOAT);
            if (descr == NULL) {
                return -1;  // caught by PyErr_Occurred on return
            }
            npy_float out = 0;
            PyArray_CastScalarToCtype(obj, &out, outcode);
            DECREF(outcode);
            return out;
        }

        else {

            // not an array scalar
            PyErr_SetString(PyExc_TypeError, "Unsupported type, expected numpy default float.");
            return -1;
        }

    }
}


npy_double unwrap_double(PyObject* obj) {
    if (obj == NULL) {
        return -1;
    }

    if (PyArray_IsPythonNumber(obj)) {
        // disambiguate type
        if (PyFloat_CheckExact(obj)) {
            npy_double out = PyFloat_AsDouble(obj);
            return out;
        }

        else if (PyLong_Check(obj)) {
            npy_double out = PyLong_AsDouble(obj);
            return out;
        }

        else {  // don't coerce any other builtin type
            PyErr_SetString(PyExc_TypeError, "Unsupported numerical python type for expected type float or np.float32.\n")
            return -1;
        }
    }

    else {
        // Check for exact numpy type
        if (PyArray_CheckScalar(obj)) {
            PyArray_Descr* outcode = PyArray_DescrFromType(NPY_DOUBLE);
            if (descr == NULL) {
                return -1;  // caught by PyErr_Occurred on return
            }
            npy_double out = 0;
            PyArray_CastScalarToCtype(obj, &out, outcode);
            DECREF(outcode);
            return out;
        }

        else {

            // not an array scalar
            PyErr_SetString(PyExc_TypeError, "Unsupported type, expected numpy double.");
            return -1;
        }

    }

}


npy_int unwrap_int(PyObject* obj) {
    if (obj == NULL) {
        return -1;
    }

    if (PyArray_IsPythonNumber(obj)) {
        // disambiguate type
        if (PyFloat_CheckExact(obj)) {
            npy_int out = (npy_int)PyFloat_AsDouble(obj);
            return (npy_int) out;
        }

        else if (PyLong_Check(obj)) {
            npy_int out = (npy_int)PyLong_AsLong(obj);
            return out;
        }

        else {  // don't coerce any other builtin type
            PyErr_SetString(PyExc_TypeError, "Unsupported numerical python type for expected type float or np.float32.\n")
            return -1;
        }
    }

    else {
        // Check for exact numpy type
        if (PyArray_CheckScalar(obj)) {
            PyArray_Descr* outcode = PyArray_DescrFromType(NPY_INT);
            if (descr == NULL) {
                return -1;  // caught by PyErr_Occurred on return
            }
            npy_int out = 0;
            PyArray_CastScalarToCtype(obj, &out, outcode);
            DECREF(outcode);
            return out;
        }

        else {
            PyErr_SetString(PyExc_TypeError, "Unsupported type, expected numpy int.");
            return -1;
        }

    }
}


npy_longlong unwrap_longlong(PyObject* obj) {
    if (obj == NULL) {
        return -1;
    }

    if (PyArray_IsPythonNumber(obj)) {
        // disambiguate type
        if (PyFloat_CheckExact(obj)) {
            npy_longlong out = (npy_longlong)PyFloat_AsDouble(obj);
            return out;
        }

        else if (PyLong_Check(obj)) {
            npy_longlong out = (npy_longlong)PyLong_AsDouble(obj);
            return out;
        }

        else {  // don't coerce any other builtin type
            PyErr_SetString(PyExc_TypeError, "Unsupported numerical python type for expected type long long.\n")
                return -1;
        }
    }

    else {
        // Check for exact numpy type
        if (PyArray_CheckScalar(obj)) {
            PyArray_Descr* outcode = PyArray_DescrFromType(NPY_LONGLONG);
            if (descr == NULL) {
                return -1;  // caught by PyErr_Occurred on return
            }
            npy_double out = 0;
            PyArray_CastScalarToCtype(obj, &out, outcode);
            DECREF(outcode);
            return out;
        }

        else {

            PyErr_SetString(PyExc_TypeError, "Unsupported type, expected numpy long long.");
            return -1;
        }

    }
}


npy_bool unwrap_bool(PyObject* obj) {
    // check PyBool_Check
    // compare to Py_False and Py_True
    // to determine what
    if (obj == NULL) {
        return -1;
    }

    if (PyBool_Check(obj)) {
        // check if python bool, otherwise fail
        if (obj == Py_True) {
            return true;
        }
        else {
            return false;
        }
    }

    else {
        // Check for exact numpy type
        if (PyArray_CheckScalar(obj)) {
            PyArray_Descr* outcode = PyArray_DescrFromType(NPY_BOOL);
            if (descr == NULL) {
                return False;  // caught by PyErr_Occurred on return
            }
            npy_bool out = 0;
            PyArray_CastScalarToCtype(obj, &out, outcode);
            DECREF(outcode);
            return out;
        }
        else {
            PyErr_SetString(PyExc_TypeError, "Unsupported type, expected bool.");
            return False;
        }
    }
}
