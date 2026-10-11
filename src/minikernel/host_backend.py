"""Run generated CUDA C++ on the CPU by stripping CUDA keywords.

This is the no-GPU differential path: interpreter vs clang++ + UBSan.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from .codegen_cuda import compile_to_cuda
from .errors import CompileError, LaunchError
from .interpreter import POISON
from .nodes import Kernel

RUNTIME = Path(__file__).resolve().parents[2] / "runtime"


def run_cuda_on_cpu(
    kernel: Kernel,
    inputs: dict[str, list[int]],
    n: int,
    block_size: int = 32,
    opt: str = "-O0",
) -> dict[str, list[int]]:
    cxx = os.environ.get("CXX", "clang++")
    code = '#include "cuda_host_shim.hpp"\n' + compile_to_cuda(kernel)
    arrays = []
    for name in kernel.inputs:
        vals = ", ".join("INT_MIN" if v == -2**31 else f"(int){v}" for v in inputs[name])
        arrays.append(f"static int p_{name}[] = {{{vals}}};")
    for name in kernel.outputs:
        arrays.append(f"static int p_{name}[{n}];")
    args = ", ".join([f"p_{a}" for a in kernel.inputs + kernel.outputs] + [f"{n}u"])
    prints = "".join(
        f'for (int i = 0; i < {n}; i++) printf("%d ", p_{o}[i]); printf("\\n");'
        for o in kernel.outputs
    )
    inits = "".join(f"for (int i = 0; i < {n}; i++) p_{o}[i] = {POISON};" for o in kernel.outputs)
    code += (
        "\n#include <cstdio>\n"
        + "\n".join(arrays)
        + f"""
int main() {{
    {inits}
    blockDim.x = {block_size};
    for (unsigned t = 0; t < {n}; t++) {{
        blockIdx.x = t / {block_size};
        threadIdx.x = t % {block_size};
        {kernel.name}({args});
    }}
    {prints}
    return 0;
}}
"""
    )

    with tempfile.TemporaryDirectory(prefix="minikernel_") as tmp:
        src = Path(tmp) / "cpu_test.cpp"
        exe = Path(tmp) / "cpu_test"
        src.write_text(code)
        compile_cmd = [
            cxx,
            "-std=c++17",
            opt,
            "-Wno-unused-variable",
            "-Wno-unused-but-set-variable",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            "-I",
            str(RUNTIME),
            str(src),
            "-o",
            str(exe),
        ]
        compiled = subprocess.run(compile_cmd, capture_output=True, text=True)
        if compiled.returncode != 0:
            raise CompileError(compiled.stderr or compiled.stdout or "host compile failed")
        ran = subprocess.run([str(exe)], capture_output=True, text=True)
        if ran.returncode != 0 or ran.stderr:
            raise LaunchError(ran.stderr or f"host program exited {ran.returncode}")
        lines = [line for line in ran.stdout.strip().splitlines() if line.strip()]
        if len(lines) != len(kernel.outputs):
            raise LaunchError(f"expected {len(kernel.outputs)} output lines, got {len(lines)}")
        result = {}
        for name, line in zip(kernel.outputs, lines):
            result[name] = [int(x) for x in line.split()]
        return result
