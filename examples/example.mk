// The running example from the spec notes.
kernel example(in a: int[], in b: int[], out c: int[]) {
    let x = a[tid] * 3;
    let y = b[tid + 1];
    let acc = 0;
    for i in 0..4 {
        acc = acc + ((x >> i) ^ y);
    }
    if (acc > 100) {
        c[tid] = acc / (y - 7);
    } else {
        c[tid] = -acc;
    }
}
