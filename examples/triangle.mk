// c[i] = 0 + 1 + ... + i
kernel triangle(out c: int[]) {
    let acc = 0;
    for k in 0..64 {
        if (k <= tid) {
            acc = acc + k;
        }
    }
    c[tid] = acc;
}
