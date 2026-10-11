# MiniKernel Language Specification v0.1

## 1. Introduction

MiniKernel is a small language for writing GPU kernels. Its purpose is compiler testing: **every program that passes the checker has exactly one correct output** for a given input.

There is no undefined behavior, no data races, and no unspecified evaluation order. If two implementations disagree, at least one of them is wrong.

A MiniKernel program is one kernel. The test harness launches it with `n` threads. Thread `tid` has index `0 .. n-1`. Every array has length `n`.

## 2. Lexical structure

Tokens:

- Keywords: `kernel`, `in`, `out`, `int`, `let`, `if`, `else`, `for`, `tid`
- Identifiers: letter or `_`, then letters, digits, or `_`
- Integer literals: one or more digits. The value is taken modulo 2³² and then interpreted as a signed 32-bit integer (so the token `2147483648` means −2147483648).
- Comments: `//` to end of line
- Operators: `+ - * / % & | ^ << >> == != < <= > >= && || ! ~`
- Punctuation: `( ) { } [ ] , : ; = ..`

Whitespace is ignored except that it separates tokens.

## 3. Grammar

```
program    → kernel IDENT "(" param ( "," param )* ")" block
param      → ( "in" | "out" ) IDENT ":" "int" "[" "]"
block      → "{" statement* "}"

statement  → "let" IDENT "=" expr ";"
           | IDENT "=" expr ";"
           | IDENT "[" "tid" "]" "=" expr ";"
           | "if" "(" expr ")" block ( "else" block )?
           | "for" IDENT "in" const ".." const block

const      → "-"? NUMBER
expr       → or
or         → and ( "||" and )*
and        → equality ( "&&" equality )*
equality   → comparison ( ( "==" | "!=" ) comparison )*
comparison → bitor ( ( "<" | "<=" | ">" | ">=" ) bitor )*
bitor      → bitxor ( "|" bitxor )*
bitxor     → bitand ( "^" bitand )*
bitand     → shift ( "&" shift )*
shift      → term ( ( "<<" | ">>" ) term )*
term       → factor ( ( "+" | "-" ) factor )*
factor     → unary ( ( "*" | "/" | "%" ) unary )*
unary      → ( "-" | "!" | "~" ) unary | primary
primary    → NUMBER | "tid" | IDENT | IDENT "[" expr "]" | "(" expr ")"
```

A kernel must declare at least one `out` array.

## 4. Types

The only type is `int`: a 32-bit two’s-complement integer.

`int[]` is an array of `int`. Parameters marked `in` are read-only. Parameters marked `out` are write-only. A name cannot be both.

Every array has length `n`, the number of threads (see §7).

## 5. Expressions

Every operator is defined for every pair of `int` values. Overflow wraps. Results of comparisons and logic are `1` (true) or `0` (false). Any non-zero value is true.

Expressions have no side effects. Evaluation order cannot change the result, so it is left unspecified.

### 5.1 Arithmetic

| Operator | Result |
|---|---|
| `a + b`, `a - b`, `a * b` | Wrap to 32 bits (two’s complement) |
| `-a` | Wrap; so `-(-2147483648)` is `-2147483648` |
| `a / b` | If `b` is 0, **0**. If `a` is −2147483648 and `b` is −1, **−2147483648**. Otherwise the mathematical quotient rounded toward zero. |
| `a % b` | If `b` is 0, **`a`**. If `a` is −2147483648 and `b` is −1, **0**. Otherwise `a - b * (a / b)`. |

These rules keep `(a / b) * b + (a % b) == a` for every `a` and `b`.

### 5.2 Shifts

`a << b` is `wrap32(a << (b & 31))`.  
`a >> b` is an **arithmetic** right shift of `a` by `(b & 31)` bits.  
So `1 << 33` is `1 << 1` = 2, and `-8 >> 1` is −4.

### 5.3 Bitwise and logic

`& | ^ ~` are the usual 32-bit bitwise operations.  
`!a` is 1 if `a` is 0, else 0.  
`a && b` is 1 if both are non-zero.  
`a || b` is 1 if at least one is non-zero.

Both sides of `&&` and `||` may be evaluated; that cannot change the result.

### 5.4 Indexing

`a[i]` is legal only when `a` is an `in` array.  
If `0 <= i < n`, the result is `a[i]`. Otherwise the result is **0**. Negative indexes therefore read as 0.

### 5.5 `tid`

`tid` is this thread’s index, an `int` in `0 .. n-1`.

## 6. Statements

- `let x = e;` declares `x` in the current block and sets it to `e`. `e` is evaluated before `x` becomes visible, so `let x = x + 1;` is an error unless an outer `x` exists.
- `x = e;` assigns to an existing variable. Assigning a `for` loop variable is an error.
- `out[tid] = e;` is the only store. The index must be the keyword `tid`. Writing an `in` array is an error.
- `if (e) { ... } else { ... }` runs one block. The `else` part may be omitted.
- `for i in A..B { ... }` runs the body with `i = A, A+1, ..., B-1`. `A` and `B` are integer constants. If `B <= A` the body runs zero times. `i` cannot be assigned in the body.

Blocks `{ }` open a new scope. Inner names may shadow outer locals, but may not shadow kernel parameters.

## 7. Kernels and threads

The harness chooses `n >= 1` and launches at least `n` threads. Threads with `tid >= n` do nothing.

A thread may only write `out[tid]` for `out` arrays it declared. Combined with read-only inputs, this makes data races impossible.

Unwritten output elements have the poison value `0x7EADBEEF` (`2126032623`) so a missing store is visible.

## 8. Static rules (compile-time errors)

The checker must reject a program if:

1. A variable is used before `let` (or outside its block).
2. A name is declared twice in the same block.
3. A kernel parameter is shadowed by `let`.
4. A loop variable is assigned.
5. A store writes anything other than an `out` array at `tid`.
6. An index reads anything other than an `in` array.
7. A parameter name is duplicated, or is both `in` and `out`.

## 9. Notes for implementers

### 9.1 PTX back end

- Calling convention: one `.u64` pointer per `in` array (declaration order), then one per `out` array, then `.u32 n_param`.
- Header: `.version 7.0`, `.target sm_75`, `.address_size 64`.
- `add.s32` / `sub.s32` / `mul.lo.s32` wrap; they match §5.1.
- `div.s32` / `rem.s32` are unspecified for a zero divisor. Never issue them with `b == 0` or with `INT_MIN / -1`. Replace the divisor with 1 and then select the spec result.
- `shl` / `shr` should still mask the count with 31 so they match §5.2 (PTX clamps large counts; the spec masks).
- Out-of-range loads must not touch global memory; predicate the `ld.global` and return 0.

### 9.2 CUDA C++ back end

- Emit helpers that do arithmetic on `unsigned` and convert back to `int` so signed overflow is never UB.
- Guard `/` and `%` as in §5.1 before using C++ `/` and `%`.
- `>>` on a negative `int` is implementation-defined in C++17. MiniKernel assumes an arithmetic shift, which clang, gcc, and nvcc all provide.
- Converting an out-of-range `unsigned` to `int` is implementation-defined in C++17; the same compilers wrap. Documented here on purpose.
- Use `extern "C"` so the kernel name is not mangled.

### 9.3 Host differential path

When no GPU is available, the generated CUDA is compiled as host C++ (CUDA keywords defined away) with `-fsanitize=undefined` and compared to the interpreter. A sanitizer report means the CUDA back end emitted UB.
