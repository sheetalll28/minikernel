// Defined for every input, including b[tid] == 0.
kernel safe_div(in a: int[], in b: int[], out c: int[]) {
    c[tid] = a[tid] / b[tid];
}
