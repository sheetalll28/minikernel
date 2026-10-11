from minikernel.frontend import compile_source
from minikernel.interpreter import POISON, interpret


def test_add_one():
    k = compile_source("kernel k(in a: int[], out c: int[]) { c[tid] = a[tid] + 1; }")
    out = interpret(k, {"a": [10, 20, 30]}, 3)
    assert out["c"] == [11, 21, 31]


def test_oob_read_is_zero():
    k = compile_source("kernel k(in a: int[], out c: int[]) { c[tid] = a[tid + 100]; }")
    out = interpret(k, {"a": [5, 6]}, 2)
    assert out["c"] == [0, 0]


def test_missing_store_is_poison():
    k = compile_source("kernel k(out c: int[]) { let x = 1; }")
    out = interpret(k, {}, 2)
    assert out["c"] == [POISON, POISON]


def test_triangle_small():
    src = """
    kernel triangle(out c: int[]) {
        let acc = 0;
        for k in 0..8 {
            if (k <= tid) { acc = acc + k; }
        }
        c[tid] = acc;
    }
    """
    k = compile_source(src)
    out = interpret(k, {}, 5)
    assert out["c"] == [0, 1, 3, 6, 10]


def test_empty_loop():
    k = compile_source(
        "kernel k(out c: int[]) { let acc = 5; for i in 3..3 { acc = acc + 1; } c[tid] = acc; }"
    )
    assert interpret(k, {}, 1)["c"] == [5]
