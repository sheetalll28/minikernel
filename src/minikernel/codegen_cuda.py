from .nodes import Assign, Binary, For, If, Index, Kernel, Let, Num, Store, Tid, Unary, Var

CUDA_PRELUDE = r"""#include <climits>

__device__ __forceinline__ int mk_add(int a, int b) { return (int)((unsigned)a + (unsigned)b); }
__device__ __forceinline__ int mk_sub(int a, int b) { return (int)((unsigned)a - (unsigned)b); }
__device__ __forceinline__ int mk_mul(int a, int b) { return (int)((unsigned)a * (unsigned)b); }
__device__ __forceinline__ int mk_neg(int a) { return (int)(0u - (unsigned)a); }

__device__ __forceinline__ int mk_div(int a, int b) {
    if (b == 0) return 0;
    if (a == INT_MIN && b == -1) return INT_MIN;
    return a / b;
}

__device__ __forceinline__ int mk_mod(int a, int b) {
    if (b == 0) return a;
    if (a == INT_MIN && b == -1) return 0;
    return a % b;
}

__device__ __forceinline__ int mk_shl(int a, int b) { return (int)((unsigned)a << (b & 31)); }
__device__ __forceinline__ int mk_shr(int a, int b) { return a >> (b & 31); }

__device__ __forceinline__ int mk_load(const int *arr, int i, unsigned n) {
    return (unsigned)i < n ? arr[i] : 0;
}
"""

FUNCS = {
    "+": "mk_add",
    "-": "mk_sub",
    "*": "mk_mul",
    "/": "mk_div",
    "%": "mk_mod",
    "<<": "mk_shl",
    ">>": "mk_shr",
}


class CudaGenerator:
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.lines: list[str] = []

    def emit(self, line: str, depth: int) -> None:
        self.lines.append("    " * depth + line)

    def generate(self) -> str:
        k = self.kernel
        params = [f"const int *p_{a}" for a in k.inputs] + [f"int *p_{a}" for a in k.outputs]
        params.append("unsigned n")
        self.lines.append(f'extern "C" __global__ void {k.name}({", ".join(params)}) {{')
        self.emit("int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);", 1)
        self.emit("if ((unsigned)tid >= n) return;", 1)
        for stmt in k.body:
            self.stmt(stmt, 1)
        self.lines.append("}")
        return CUDA_PRELUDE + "\n" + "\n".join(self.lines) + "\n"

    def stmt(self, node, depth: int) -> None:
        if isinstance(node, Let):
            self.emit(f"int v_{node.name} = {self.expr(node.value)};", depth)
        elif isinstance(node, Assign):
            self.emit(f"v_{node.name} = {self.expr(node.value)};", depth)
        elif isinstance(node, Store):
            self.emit(f"p_{node.array}[tid] = {self.expr(node.value)};", depth)
        elif isinstance(node, If):
            self.emit(f"if ({self.expr(node.cond)}) {{", depth)
            for s in node.then_body:
                self.stmt(s, depth + 1)
            self.emit("} else {", depth)
            for s in node.else_body:
                self.stmt(s, depth + 1)
            self.emit("}", depth)
        elif isinstance(node, For):
            v = f"v_{node.var}"
            self.emit(f"for (int {v} = {node.start}; {v} < {node.end}; {v}++) {{", depth)
            for s in node.body:
                self.stmt(s, depth + 1)
            self.emit("}", depth)
        else:
            raise TypeError(f"not a statement: {node!r}")

    def expr(self, node) -> str:
        if isinstance(node, Num):
            return "INT_MIN" if node.value == -2**31 else f"({node.value})"
        if isinstance(node, Var):
            return f"v_{node.name}"
        if isinstance(node, Tid):
            return "tid"
        if isinstance(node, Index):
            return f"mk_load(p_{node.array}, {self.expr(node.index)}, n)"
        if isinstance(node, Unary):
            x = self.expr(node.operand)
            return {"-": f"mk_neg({x})", "!": f"((int)!{x})", "~": f"(~{x})"}[node.op]
        if isinstance(node, Binary):
            a, b = self.expr(node.left), self.expr(node.right)
            if node.op in FUNCS:
                return f"{FUNCS[node.op]}({a}, {b})"
            if node.op in ("&&", "||"):
                return f"((int)({a} != 0 {node.op} {b} != 0))"
            return f"((int)({a} {node.op} {b}))"
        raise TypeError(f"not an expression: {node!r}")


def compile_to_cuda(kernel: Kernel) -> str:
    return CudaGenerator(kernel).generate()
