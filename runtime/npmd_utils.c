#include "npmd_utils.h"


int dim_counts_match(PyArrayObject** arrs, int count){
    if(count < 1){
       return -1;
    }
    if(count < 2){
       return 1;
    }
    PyArrayObject* first = arrs[0];
    int dims = PyArray_NDIM(first);
    // check for identical dim count
    for(int i = 1; i < count; ++i){
        if(dims != PyArray_NDIM(arrs[i])){
            return 0;
	}
    }
    return 1;
}


int dims_match(PyArrayObject** arrs, int count){
    if(count < 1){
       return -1;
    }
    else if(count < 2){
        return 1;
    }
    if(!dim_counts_match(arrs, count)){
        return 0;
    }
    for(int i = 1; i < count; ++i){
        for(int j = 0; j < dims; ++j){
            int n = PyArray_DIM(first, j)
            for(k = 1; k < count; ++k){
                if(n != PyArray_DIM(arrs[k], j)){
                    return 0;
                }
            }
	}
    }
    return 1;
}


int max_dim_count(PyArrayObject** arrays, int count){
    if(count < 1){
        return -1;
    }
    int max_dims = PyArray_NDIM(arrays[0]);
    for(int i = 1; i < count; ++i){
        int dims = PyArray_NDIM(arrays[i]);
	if(dims > max_dims){
            max_dims = dims;
	}
    }
}


int min_dim_count(PyArrayObject** arrays, int count){
    if(count < 1){
        return -1;
    }
    int min_dims = PyArray_NDIM(arrays[0]);
    for(int i = 1; i < count; ++i){
        int dims = PyArray_NDIM(arrays[i]);
	if(dims < min_dims){
            min_dims = dims;
	}
    }
    return min_dims;
}


int can_simple_broadcast(PyArrayObject* left, PyArrayObject* right){
    if(left == NULL){
        return -1;
    }
    else if (right == NULL){
        return -1;
    }
    int left_ndims = PyArray_NDIM(left);
    int right_ndims = PyArray_NDIM(right);
    int lim = left_ndims >= right_ndims ? left_ndims : right_ndims;
    for(int i = 1; i <= right_ndims; ++i){
        npy_intp dim_left = PyArray_DIM(left, left_ndims - i);
        npy_intp dim_right = PyArray_DIM(right, right_ndims - i);
        assert(dim_left != NULL);
	assert(dim_right != NULL);
	if(*dim_left != *dim_right){
            return 0;
	}
    }
}


int dims_match_left(PyArrayObject* left, PyArrayObject* right){
    if(left == NULL || right == NULL){
        return -1;
    }
    return PyArray_NDIM(left) >= PyArray_NDIM(right);
}


int dims_match_right(PyArrayObject* left, PyArrayObject* right){
    if(left == NULL || right == NULL){
        return -1;
    }
    return PyArray_NDIM(right) >= PyArray_NDIM(left);
}

