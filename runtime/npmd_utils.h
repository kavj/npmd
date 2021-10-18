#ifndef NPMD_UTILS_H
#define NPMD_UTILS_H

#include<numpy/ndarraytypes.h>
#include <stdbool.h>


bool dim_counts_match(PyArrayObject** arrs, int count);

bool dims_match(PyArrayObject** arrs, int count);

int max_dim_count(PyArrayObject** arrays, int count);

int min_dim_count(PyArrayObject** arrays, int count);

bool can_simple_broadcast(PyArrayObject* left, PyArrayObject* right);

bool dims_match_left(PyArrayObject* left, PyArrayObject* right);

bool dims_match_right(PyArrayObject* left, PyArrayObject* right);

#endif

