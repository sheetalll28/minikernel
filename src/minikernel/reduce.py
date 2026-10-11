from .nodes import Binary, For, If, Index, Kernel, Num, Unary, copy_kernel, walk


def ddmin(data: list, is_interesting, granularity: int = 2) -> list:
    """Zeller's ddmin: return a 1-minimal subset that stays interesting."""
    if not data:
        return data
    assert is_interesting(data), "the original input must already be interesting"
    n = granularity
    while len(data) >= 2:
        start = 0
        chunk = max(1, len(data) // n)
        reduced = False
        while start < len(data):
            complement = data[:start] + data[start + chunk :]
            if complement and is_interesting(complement):
                data = complement
                n = max(2, n - 1)
                reduced = True
                break
            start += chunk
        if not reduced:
            if n >= len(data):
                break
            n = min(len(data), n * 2)
    return data


def reduce_kernel(kernel: Kernel, interesting) -> Kernel:
    """Shrink a kernel body, then try simple tree edits."""

    def stmt_pred(body):
        return interesting(copy_kernel(kernel, body))

    body = list(kernel.body)
    if stmt_pred(body):
        body = ddmin(body, stmt_pred)
    current = copy_kernel(kernel, body)
    current = _shrink_control(current, interesting)
    current = _shrink_exprs(current, interesting)
    return current


def _shrink_control(kernel: Kernel, interesting) -> Kernel:
    changed = True
    while changed:
        changed = False
        for i, stmt in enumerate(kernel.body):
            if isinstance(stmt, If):
                for candidate in (stmt.then_body, stmt.else_body):
                    trial = copy_kernel(kernel, kernel.body[:i] + candidate + kernel.body[i + 1 :])
                    if interesting(trial):
                        kernel = trial
                        changed = True
                        break
            elif isinstance(stmt, For) and stmt.body:
                trial = copy_kernel(kernel, kernel.body[:i] + stmt.body + kernel.body[i + 1 :])
                if interesting(trial):
                    kernel = trial
                    changed = True
            if changed:
                break
    return kernel


def _replace_first(node, target, replacement):
    if node is target:
        return replacement
    if isinstance(node, Unary):
        return Unary(node.op, _replace_first(node.operand, target, replacement))
    if isinstance(node, Binary):
        return Binary(
            node.op,
            _replace_first(node.left, target, replacement),
            _replace_first(node.right, target, replacement),
        )
    if isinstance(node, Index):
        return Index(node.array, _replace_first(node.index, target, replacement), node.line)
    if hasattr(node, "value") and hasattr(node, "name") and hasattr(node, "line"):
        # Let / Assign / Store
        from .nodes import Assign, Let, Store

        inner = _replace_first(node.value, target, replacement)
        if isinstance(node, Let):
            return Let(node.name, inner, node.line)
        if isinstance(node, Assign):
            return Assign(node.name, inner, node.line)
        if isinstance(node, Store):
            return Store(node.array, inner, node.line)
    if isinstance(node, If):
        return If(
            _replace_first(node.cond, target, replacement),
            [_replace_first(s, target, replacement) for s in node.then_body],
            [_replace_first(s, target, replacement) for s in node.else_body],
        )
    if isinstance(node, For):
        return For(
            node.var,
            node.start,
            node.end,
            [_replace_first(s, target, replacement) for s in node.body],
            node.line,
        )
    return node


def _shrink_exprs(kernel: Kernel, interesting) -> Kernel:
    current = kernel
    progress = True
    while progress:
        progress = False
        for node in walk(current):
            candidates = []
            if isinstance(node, Binary):
                candidates = [node.left, node.right, Num(0)]
            elif isinstance(node, Unary):
                candidates = [node.operand, Num(0)]
            elif isinstance(node, Index):
                candidates = [Num(0)]
            for cand in candidates:
                trial_body = [_replace_first(s, node, cand) for s in current.body]
                trial = copy_kernel(current, trial_body)
                if interesting(trial):
                    current = trial
                    progress = True
                    break
            if progress:
                break
    return current
