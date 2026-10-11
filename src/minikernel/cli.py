from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .codegen_cuda import compile_to_cuda
from .codegen_ptx import compile_to_ptx
from .frontend import compile_file, compile_source
from .generate import generate_inputs
from .harness import fuzz, run_case, signature
from .interpreter import interpret
from .pretty import kernel_src
from .reduce import ddmin, reduce_kernel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minikernel",
        description="Compile MiniKernel programs, run the interpreter, or fuzz the back ends.",
    )
    parser.add_argument("--version", action="version", version=f"minikernel {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run a kernel with the reference interpreter")
    run.add_argument("source", type=Path)
    run.add_argument("--n", type=int, default=8, help="threads / array length")
    run.add_argument("--input", action="append", default=[], metavar="name=1,2,3")

    compile_p = sub.add_parser("compile", help="emit PTX or CUDA C++")
    compile_p.add_argument("source", type=Path)
    compile_p.add_argument("--target", choices=("ptx", "cuda"), default="ptx")
    compile_p.add_argument("-o", "--output", type=Path)

    fuzz_p = sub.add_parser("fuzz", help="generate random kernels and compare implementations")
    fuzz_p.add_argument("--count", type=int, default=20)
    fuzz_p.add_argument("--seed", type=int, default=0)
    fuzz_p.add_argument("--n", type=int, default=8)
    fuzz_p.add_argument("--depth", type=int, default=2)

    check_p = sub.add_parser("check", help="parse and type-check a kernel")
    check_p.add_argument("source", type=Path)

    reduce_p = sub.add_parser("reduce", help="demo ddmin, or shrink a failing seed")
    reduce_p.add_argument("--demo", action="store_true")
    reduce_p.add_argument("--seed", type=int)
    reduce_p.add_argument("--n", type=int, default=8)
    return parser


def parse_inputs(raw: list[str], names: list[str], n: int) -> dict[str, list[int]]:
    supplied = {}
    for item in raw:
        if "=" not in item:
            raise SystemExit(f"expected name=1,2,3, got {item!r}")
        name, values = item.split("=", 1)
        supplied[name] = [int(v) for v in values.split(",") if v != ""]
    inputs = {}
    for i, name in enumerate(names):
        if name in supplied:
            values = supplied[name]
            if len(values) != n:
                raise SystemExit(f"input {name} has {len(values)} values, expected {n}")
            inputs[name] = values
        else:
            inputs[name] = [i + k for k in range(n)]
    return inputs


def cmd_run(args):
    kernel = compile_file(args.source)
    inputs = parse_inputs(args.input, kernel.inputs, args.n)
    outputs = interpret(kernel, inputs, args.n)
    for name, values in outputs.items():
        print(f"{name}: {' '.join(str(v) for v in values)}")


def cmd_compile(args):
    kernel = compile_file(args.source)
    text = compile_to_ptx(kernel) if args.target == "ptx" else compile_to_cuda(kernel)
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


def cmd_check(args):
    kernel = compile_file(args.source)
    print(f"ok: kernel {kernel.name} ({len(kernel.inputs)} in, {len(kernel.outputs)} out)")


def cmd_fuzz(args):
    results = fuzz(count=args.count, seed=args.seed, n=args.n, depth=args.depth)
    counts = {"PASS": 0, "MISMATCH": 0, "TOOL ERROR": 0, "INVALID": 0}
    for i, result in enumerate(results, 1):
        counts[result.kind] = counts.get(result.kind, 0) + 1
        if result.kind != "PASS":
            print(f"seed {result.seed}: {result.kind} {result.which}")
            if result.detail:
                print(f"  {result.detail[:300]}")
        if i % 5 == 0 or i == len(results):
            print(
                f"[{i}/{len(results)}] pass={counts['PASS']} "
                f"mismatch={counts['MISMATCH']} error={counts['TOOL ERROR']} invalid={counts['INVALID']}"
            )
    unique = {signature(r) for r in results if r.kind != "PASS"}
    print(f"unique non-pass signatures: {len(unique)}")
    sys.exit(0 if counts["MISMATCH"] == 0 and counts["TOOL ERROR"] == 0 else 1)


def cmd_reduce(args):
    if args.demo:
        original = [1, 7, 3, 5, 9, 2, 8]
        result = ddmin(original, lambda xs: 7 in xs and 3 in xs)
        print(result)
        return
    if args.seed is None:
        raise SystemExit("use --demo or --seed N")
    from .generate import generate_kernel

    kernel = generate_kernel(args.seed)
    inputs = generate_inputs(args.seed, args.n, kernel.inputs)

    def interesting(trial):
        outcome = run_case(trial, inputs, args.n)
        return outcome.kind == "MISMATCH"

    if not interesting(kernel):
        print("seed does not mismatch; nothing to reduce")
        return
    small = reduce_kernel(kernel, interesting)
    print(kernel_src(small), end="")


def main(argv=None):
    args = build_parser().parse_args(argv)
    commands = {
        "run": cmd_run,
        "compile": cmd_compile,
        "check": cmd_check,
        "fuzz": cmd_fuzz,
        "reduce": cmd_reduce,
    }
    commands[args.cmd](args)


if __name__ == "__main__":
    main()
