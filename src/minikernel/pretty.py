from .nodes import (
    Assign,
    Binary,
    For,
    If,
    Index,
    Kernel,
    Let,
    Num,
    Store,
    Tid,
    Unary,
    Var,
)
from .semantics import PRECEDENCE


def expr_src(node, parent_prec=0) -> str:
    if isinstance(node, Num):
        return str(node.value)
    if isinstance(node, Var):
        return node.name
    if isinstance(node, Tid):
        return "tid"
    if isinstance(node, Index):
        return f"{node.array}[{expr_src(node.index)}]"
    if isinstance(node, Unary):
        return f"{node.op}{expr_src(node.operand, 99)}"
    if isinstance(node, Binary):
        prec = PRECEDENCE[node.op]
        left = expr_src(node.left, prec)
        right = expr_src(node.right, prec + 1)
        text = f"{left} {node.op} {right}"
        if prec < parent_prec:
            return f"({text})"
        return text
    raise TypeError(f"not an expression: {node!r}")


def stmt_src(stmt, indent: int) -> list[str]:
    pad = "    " * indent
    if isinstance(stmt, Let):
        return [f"{pad}let {stmt.name} = {expr_src(stmt.value)};"]
    if isinstance(stmt, Assign):
        return [f"{pad}{stmt.name} = {expr_src(stmt.value)};"]
    if isinstance(stmt, Store):
        return [f"{pad}{stmt.array}[tid] = {expr_src(stmt.value)};"]
    if isinstance(stmt, If):
        lines = [f"{pad}if ({expr_src(stmt.cond)}) {{"]
        for inner in stmt.then_body:
            lines.extend(stmt_src(inner, indent + 1))
        if stmt.else_body:
            lines.append(f"{pad}}} else {{")
            for inner in stmt.else_body:
                lines.extend(stmt_src(inner, indent + 1))
        lines.append(f"{pad}}}")
        return lines
    if isinstance(stmt, For):
        lines = [f"{pad}for {stmt.var} in {stmt.start}..{stmt.end} {{"]
        for inner in stmt.body:
            lines.extend(stmt_src(inner, indent + 1))
        lines.append(f"{pad}}}")
        return lines
    raise TypeError(f"not a statement: {stmt!r}")


def kernel_src(kernel: Kernel) -> str:
    params = [f"in {name}: int[]" for name in kernel.inputs]
    params += [f"out {name}: int[]" for name in kernel.outputs]
    lines = [f"kernel {kernel.name}({', '.join(params)}) {{"]
    for stmt in kernel.body:
        lines.extend(stmt_src(stmt, 1))
    lines.append("}")
    return "\n".join(lines) + "\n"
