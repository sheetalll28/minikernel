# MiniKernel

A tiny GPU kernel language, a compiler that emits **NVIDIA PTX**, and a **differential tester** that compares three views of the same program.

Every valid MiniKernel program has **exactly one correct output**. There is no undefined behavior and no data races. A mismatch is a real bug in an implementation — not a disagreement about the spec.

## Why

Compiler testers (Csmith, YARPGen, CUDAsmith) spend most of their effort avoiding undefined behavior, because a program with UB can legally give two answers.

MiniKernel takes the other path: **define every operation for every input** (including `x / 0` and `INT_MIN / -1`). The generator does not need safe-math wrappers. The back ends do — they must insert guards so PTX and CUDA C++ follow the spec.

## Quick start

Needs Python 3.11+ and `clang++` (for the no-GPU differential path).

```bash
cd minikernel
python3 -m pip install -e ".[dev]"
python3 -m pytest -q
```

Run the interpreter (no GPU):

```bash
python3 -m minikernel run examples/add_one.mk --n 4 --input a=10,20,30,40
# c: 11 21 31 41
```

Compile to PTX or CUDA C++:

```bash
python3 -m minikernel compile examples/add_one.mk --target ptx
python3 -m minikernel compile examples/add_one.mk --target cuda
```

Fuzz: generate random kernels, interpret them, compile the CUDA back end as host C++ with UBSan, and compare.

```bash
python3 -m minikernel fuzz --count 20 --seed 0
```

On a machine with an NVIDIA GPU, load the generated PTX with CuPy (see `notebooks/colab_first_run.ipynb`).

## Architecture

```
 MiniKernel source
        │
        ▼
   lexer → parser → checker
        │
        ▼
      AST ───────► reference interpreter      (answer key)
        │
        ├──► PTX back end  ──► ptxas / GPU driver
        │
        └──► CUDA C++ back end ──► nvcc (GPU) or clang++ + UBSan (CPU)
```

A test is a **PASS** only when every implementation writes the same `int32` arrays.

## Spec highlights

Full rules: [`spec/SPEC.md`](spec/SPEC.md). Edge cases: [`spec/spec_examples.txt`](spec/spec_examples.txt).

| Topic | Rule |
|---|---|
| Integers | 32-bit two’s complement, wrap on overflow |
| Division | `a / 0 = 0`, `INT_MIN / -1 = INT_MIN`, toward zero otherwise |
| Remainder | `a % 0 = a`, `INT_MIN % -1 = 0` |
| Shifts | amount is `b & 31`; right shift is arithmetic |
| Arrays | out-of-range reads return 0; a thread may only write `out[tid]` |
| Loops | `for i in A..B` with constant bounds — they always terminate |

## Layout

```
spec/                 language spec and oracle examples
src/minikernel/       compiler, interpreter, generator, harness
examples/             hand-written kernels
tests/                pytest (CPU; no GPU required)
notebooks/            Colab notes for real-GPU runs
```

## Results

**CPU (this repo, no GPU)**

- `pytest -q`: 28 passed (spec examples, parser, interpreter, host CUDA + UBSan, `ddmin`)
- `minikernel fuzz --count 5 --seed 0`: 5/5 PASS (interpreter vs generated CUDA compiled with `clang++ -fsanitize=undefined`)

**GPU (Google Colab)**

- GPU: Tesla T4, compute capability 7.5
- `nvcc`: CUDA 13.0, V13.0.88
- Saved run: [`notebooks/colab_first_run.ipynb`](notebooks/colab_first_run.ipynb)

Generated PTX was loaded with CuPy and compared to the interpreter:

| Program | Result |
|---|---|
| `examples/add_one.mk` | MATCH |
| `examples/safe_div.mk` (includes `/ 0` and `INT_MIN / -1`) | MATCH |
| `examples/example.mk` (loop, `if`, shifts, divide) | MATCH |
| 10 random kernels (`generate_kernel`, seeds 0–9) | 10/10 PASS |
| `ptxas -arch=sm_75` on 3 examples + 5 random kernels | 8/8 accepted |

No NVIDIA compiler bug was claimed. These runs show the PTX back end follows the spec on a real GPU, including the cases where raw PTX `div` is unspecified.

