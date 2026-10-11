// Spec §5: every int operation is defined (wrap, /0, INT_MIN / -1, shift mask).
// Included by generated CUDA and by the CPU UBSan path.
#pragma once

#include <climits>

#ifdef __CUDACC__
#define MK_FN __device__ __forceinline__
#else
#define MK_FN inline
#endif

MK_FN int mk_add(int a, int b) { return (int)((unsigned)a + (unsigned)b); }
MK_FN int mk_sub(int a, int b) { return (int)((unsigned)a - (unsigned)b); }
MK_FN int mk_mul(int a, int b) { return (int)((unsigned)a * (unsigned)b); }
MK_FN int mk_neg(int a) { return (int)(0u - (unsigned)a); }

MK_FN int mk_div(int a, int b) {
    if (b == 0) return 0;
    if (a == INT_MIN && b == -1) return INT_MIN;
    return a / b;
}

MK_FN int mk_mod(int a, int b) {
    if (b == 0) return a;
    if (a == INT_MIN && b == -1) return 0;
    return a % b;
}

MK_FN int mk_shl(int a, int b) { return (int)((unsigned)a << (b & 31)); }
MK_FN int mk_shr(int a, int b) { return a >> (b & 31); }

MK_FN int mk_load(const int *arr, int i, unsigned n) {
    return (unsigned)i < n ? arr[i] : 0;
}

#undef MK_FN
