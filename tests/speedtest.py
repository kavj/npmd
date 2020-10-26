from numba import jit
import numpy as np
import time

#x = np.arange(100).reshape(10, 10)

@jit(nopython=True)
def go_fast(a,b): # Function is compiled and runs in machine code
    a *= b
#    trace = 0.0
#    for i in range(a.shape[0]):
#        trace += np.tanh(a[i, i])
#    return a + trace

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!


a = np.random.randn(2**18)
b = np.random.randn(2**18)


start = time.time()
go_fast(a,b)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
go_fast(a,b)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))


base_start = time.time()

# Numba inlines this. either way one call
a *= b

base_end = time.time()

print("baseline time = %s " % (base_end - base_start))


print("numba/base = %s " % ((end - start) / (base_end - base_start)))
