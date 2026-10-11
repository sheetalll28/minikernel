from .nodes import Assign, Binary, For, If, Index, Kernel, Let, Num, Store, Tid, Unary, Var
from .semantics import BINARY, UNARY, wrap32

POISON = 0x7EADBEEF


def interpret(kernel: Kernel, inputs: dict[str, list[int]], n: int) -> dict[str, list[int]]:
    """Run one thread per index 0..n-1. Every array has length n (SPEC §7)."""
    for name in kernel.inputs:
        if name not in inputs:
            raise KeyError(f"missing input array '{name}'")
        if len(inputs[name]) != n:
            raise ValueError(f"input '{name}' has length {len(inputs[name])}, expected {n}")

    outputs = {name: [POISON] * n for name in kernel.outputs}

    for tid in range(n):
        scopes: list[dict[str, int]] = [{}]

        def look(name: str) -> dict[str, int]:
            for scope in reversed(scopes):
                if name in scope:
                    return scope
            raise KeyError(f"undefined variable '{name}'")

        def ev(node) -> int:
            if isinstance(node, Num):
                return wrap32(node.value)
            if isinstance(node, Var):
                return look(node.name)[node.name]
            if isinstance(node, Tid):
                return tid
            if isinstance(node, Index):
                i = ev(node.index)
                if 0 <= i < n:
                    return inputs[node.array][i]
                return 0
            if isinstance(node, Unary):
                return UNARY[node.op](ev(node.operand))
            if isinstance(node, Binary):
                return BINARY[node.op](ev(node.left), ev(node.right))
            raise TypeError(f"not an expression: {node!r}")

        def block(stmts):
            scopes.append({})
            for stmt in stmts:
                ex(stmt)
            scopes.pop()

        def ex(stmt):
            if isinstance(stmt, Let):
                scopes[-1][stmt.name] = ev(stmt.value)
            elif isinstance(stmt, Assign):
                value = ev(stmt.value)
                look(stmt.name)[stmt.name] = value
            elif isinstance(stmt, Store):
                outputs[stmt.array][tid] = ev(stmt.value)
            elif isinstance(stmt, If):
                block(stmt.then_body if ev(stmt.cond) != 0 else stmt.else_body)
            elif isinstance(stmt, For):
                for i in range(stmt.start, stmt.end):
                    scopes.append({stmt.var: wrap32(i)})
                    block(stmt.body)
                    scopes.pop()
            else:
                raise TypeError(f"not a statement: {stmt!r}")

        block(kernel.body)
    return outputs
