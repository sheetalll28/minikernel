from minikernel.codegen_cuda import compile_to_cuda
from minikernel.codegen_ptx import compile_to_ptx
from minikernel.frontend import compile_file, compile_source
from minikernel.generate import generate_inputs
from minikernel.harness import run_case
from minikernel.host_backend import run_cuda_on_cpu
from minikernel.interpreter import interpret
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_ptx_has_entry_and_guards():
    k = compile_source("kernel k(in a: int[], out c: int[]) { c[tid] = a[tid] / 0; }")
    ptx = compile_to_ptx(k)
    assert ".visible .entry k(" in ptx
    assert "div.s32" in ptx
    assert "selp.s32" in ptx
    assert "ret;" in ptx


def test_cuda_uses_helpers():
    k = compile_source("kernel k(in a: int[], out c: int[]) { c[tid] = a[tid] + 1; }")
    cuda = compile_to_cuda(k)
    assert "mk_add" in cuda
    assert '#include "mk_ops.hpp"' in cuda
    assert 'extern "C"' in cuda


def test_add_one_host_matches():
    k = compile_file(EXAMPLES / "add_one.mk")
    inputs = {"a": [0, 1, -1, 2147483647]}
    n = 4
    ref = interpret(k, inputs, n)
    host = run_cuda_on_cpu(k, inputs, n)
    assert host == ref


def test_safe_div_host_matches():
    k = compile_file(EXAMPLES / "safe_div.mk")
    inputs = {"a": [7, -7, -2147483648, 5], "b": [2, 0, -1, 0]}
    ref = interpret(k, inputs, 4)
    host = run_cuda_on_cpu(k, inputs, 4)
    assert host == ref
    assert ref["c"] == [3, 0, -2147483648, 0]


def test_example_kernel_host_matches():
    k = compile_file(EXAMPLES / "example.mk")
    n = 8
    inputs = generate_inputs(1, n, k.inputs)
    outcome = run_case(k, inputs, n)
    assert outcome.kind == "PASS", outcome.detail
