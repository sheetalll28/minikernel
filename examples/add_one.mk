// Add one to every input element.
kernel add_one(in a: int[], out c: int[]) {
    c[tid] = a[tid] + 1;
}
