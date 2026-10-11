#include "mk_ops.hpp"

extern "C" __global__ void add_one(const int *p_a, int *p_c, unsigned n) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if ((unsigned)tid >= n) return;
    p_c[tid] = mk_add(mk_load(p_a, tid, n), (1));
}
