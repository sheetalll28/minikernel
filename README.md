# MiniKernel

Small language for GPU kernels, plus a compiler that emits PTX (and CUDA C++). I use it to test the compiler: run the same program in a Python interpreter and on a GPU, and check that the outputs match.

The language is specified so every valid program has one result. Overflow wraps. `a / 0` is 0. `INT_MIN / -1` is `INT_MIN`. A thread can only write `out[tid]`, so two threads do not write the same cell.

## Run (no GPU)

Python 3.11+ and `clang++`:

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest -q
python3 -m minikernel run examples/add_one.mk --n 4 --input a=10,20,30,40
python3 -m minikernel compile examples/add_one.mk --target ptx
```

`python3 -m minikernel fuzz --count 20 --seed 0` compiles generated CUDA as host C++ with UBSan and compares it to the interpreter.

GPU steps I actually ran are in `notebooks/colab_first_run.ipynb`.

## How it is structured

```
.mk source → lexer → parser → checker → AST
                                      → interpreter
                                      → PTX   (run on GPU with CuPy / ptxas)
                                      → CUDA C++ (nvcc, or clang++ on CPU)
```

Spec: `spec/SPEC.md`. Examples: `examples/`. Tests: `tests/`.

## What I actually ran

CPU, this machine:

- `pytest -q` — 28 passed
- `fuzz --count 5 --seed 0` — 5/5 PASS (interpreter vs clang++ / UBSan)

GPU, Google Colab (Tesla T4, sm_75, CUDA 13.0 V13.0.88), notebook saved with outputs:

| What | Result |
|---|---|
| `add_one.mk` | interpreter = GPU |
| `safe_div.mk` (`/ 0`, `INT_MIN / -1`) | interpreter = GPU |
| `example.mk` | interpreter = GPU |
| 10 random kernels, seeds 0–9 | 10/10 |
| `ptxas -arch=sm_75` on 3 examples + 5 random | 8/8 accepted |

I have not compared against `nvcc` on the GPU yet, and I have not found (or claimed) a bug in NVIDIA’s compiler. Sample size is small.

## Limits

- Integers only; no floats
- All arrays have the same length `n`
- Random tests so far: 10 kernels on GPU, 5 on the CPU fuzz path
- Reducer (`ddmin`) exists but I have not published a reduced GPU failure yet — nothing failed
- PTX is generated; register allocation is left to `ptxas`

## License

MIT. See `LICENSE`.
