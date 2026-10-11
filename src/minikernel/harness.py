from __future__ import annotations

from dataclasses import dataclass

from .codegen_ptx import compile_to_ptx
from .checker import check
from .errors import CompileError, LaunchError, MiniKernelError
from .generate import generate_inputs, generate_kernel
from .host_backend import run_cuda_on_cpu
from .interpreter import interpret
from .nodes import Binary, Kernel, walk


@dataclass
class Outcome:
    seed: int
    kind: str  # PASS, MISMATCH, TOOL ERROR, INVALID
    which: str
    detail: str = ""
    kernel: Kernel | None = None


def compare(expected: dict[str, list[int]], actual: dict[str, list[int]], limit: int = 8) -> list[str]:
    messages = []
    for name in expected:
        if name not in actual:
            messages.append(f"{name}: missing from actual output")
            continue
        exp, got = expected[name], actual[name]
        if len(exp) != len(got):
            messages.append(f"{name}: length {len(got)} != {len(exp)}")
            continue
        for i, (e, g) in enumerate(zip(exp, got)):
            if e != g:
                messages.append(f"{name}[{i}]: expected {e}, got {g}")
                if len(messages) >= limit:
                    return messages
    return messages


def run_case(kernel: Kernel, inputs: dict[str, list[int]], n: int) -> Outcome:
    ref = interpret(kernel, inputs, n)
    try:
        _ = compile_to_ptx(kernel)
    except Exception as exc:
        return Outcome(-1, "TOOL ERROR", "ptx-gen", str(exc), kernel)
    try:
        host = run_cuda_on_cpu(kernel, inputs, n)
    except CompileError as exc:
        return Outcome(-1, "TOOL ERROR", "cuda-host-compile", str(exc), kernel)
    except LaunchError as exc:
        return Outcome(-1, "TOOL ERROR", "cuda-host-run", str(exc), kernel)
    diffs = compare(ref, host)
    if diffs:
        return Outcome(-1, "MISMATCH", "host-cuda!=interpreter", "; ".join(diffs), kernel)
    return Outcome(-1, "PASS", "all", "", kernel)


def signature(outcome: Outcome) -> tuple:
    ops = []
    if outcome.kernel is not None:
        ops = sorted({node.op for node in walk(outcome.kernel) if isinstance(node, Binary)})
    return (outcome.kind, outcome.which, tuple(ops))


def fuzz(count: int = 20, seed: int = 0, n: int = 8, depth: int = 2) -> list[Outcome]:
    results = []
    for i in range(count):
        case_seed = seed + i
        try:
            kernel = generate_kernel(case_seed, depth=depth)
            check(kernel)
        except MiniKernelError as exc:
            results.append(Outcome(case_seed, "INVALID", "generator", str(exc)))
            continue
        except Exception as exc:
            results.append(Outcome(case_seed, "INVALID", "generator", str(exc)))
            continue
        inputs = generate_inputs(case_seed, n, kernel.inputs)
        try:
            outcome = run_case(kernel, inputs, n)
        except Exception as exc:
            results.append(Outcome(case_seed, "INVALID", "harness", str(exc), kernel))
            continue
        outcome.seed = case_seed
        results.append(outcome)
    return results
