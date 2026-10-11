// Lets generated __global__ kernels compile as ordinary C++ on the CPU.
#pragma once

#define __global__
#define __device__
#define __forceinline__ inline

struct Dim {
    unsigned x;
};

static Dim blockIdx;
static Dim blockDim;
static Dim threadIdx;
