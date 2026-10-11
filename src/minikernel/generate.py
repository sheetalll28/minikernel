import random

from .nodes import Assign, Binary, For, If, Index, Kernel, Let, Num, Store, Tid, Unary, Var
from .semantics import BINARY, UNARY

EDGE = [0, 1, -1, 2, 3, 7, -7, 31, 32, 33, -32, 2147483647, -2147483648, -2147483647, 65536, 46341]
BINOPS = list(BINARY)
UNOPS = list(UNARY)


def rand_expr(rng: random.Random, depth: int, names: list[str], inputs: list[str]):
    if depth <= 0 or rng.random() < 0.28:
        choice = rng.random()
        if choice < 0.30:
            return Num(rng.choice(EDGE))
        if choice < 0.50 and names:
            return Var(rng.choice(names))
        if choice < 0.68:
            return Tid()
        if inputs:
            idx = rng.choice(
                [
                    Tid(),
                    Binary("+", Tid(), Num(rng.choice([0, 1, -1, 5, 100]))),
                    Num(rng.choice(EDGE[:8])),
                ]
            )
            return Index(rng.choice(inputs), idx)
        return Num(rng.choice(EDGE))
    if rng.random() < 0.18:
        return Unary(rng.choice(UNOPS), rand_expr(rng, depth - 1, names, inputs))
    return Binary(
        rng.choice(BINOPS),
        rand_expr(rng, depth - 1, names, inputs),
        rand_expr(rng, depth - 1, names, inputs),
    )


def rand_block(rng, depth: int, names: list[str], inputs: list[str], outputs: list[str], counter: list[int]):
    stmts = []
    local = list(names)
    count = rng.randint(1, 3 if depth == 0 else 4)
    for _ in range(count):
        r = rng.random()
        if r < 0.40:
            counter[0] += 1
            name = f"v{counter[0]}"
            stmts.append(Let(name, rand_expr(rng, 3, local, inputs)))
            local.append(name)
        elif r < 0.55 and local:
            assignable = [v for v in local if not v.startswith("i")]
            if assignable:
                stmts.append(Assign(rng.choice(assignable), rand_expr(rng, 3, local, inputs)))
        elif r < 0.72 and depth > 0:
            then = rand_block(rng, depth - 1, local, inputs, outputs, counter)
            els = rand_block(rng, depth - 1, local, inputs, outputs, counter) if rng.random() < 0.6 else []
            stmts.append(If(rand_expr(rng, 2, local, inputs), then, els))
        elif r < 0.86 and depth > 0:
            counter[0] += 1
            iv = f"i{counter[0]}"
            start = rng.randint(-1, 2)
            body = rand_block(rng, depth - 1, local + [iv], inputs, outputs, counter)
            stmts.append(For(iv, start, start + rng.randint(0, 4), body))
        if rng.random() < 0.25 and outputs:
            stmts.append(Store(rng.choice(outputs), rand_expr(rng, 2, local, inputs)))
    return stmts


def generate_kernel(seed: int, depth: int = 2) -> Kernel:
    rng = random.Random(seed)
    inputs = ["a", "b"]
    outputs = ["c", "d"]
    counter = [0]
    body = rand_block(rng, depth, [], inputs, outputs, counter)
    body.append(Store("c", rand_expr(rng, 3, [], inputs)))
    body.append(Store("d", rand_expr(rng, 3, [], inputs)))
    return Kernel(f"k{seed}", inputs, outputs, body)


def generate_inputs(seed: int, n: int, names: list[str]) -> dict[str, list[int]]:
    rng = random.Random(seed + 1_000_003)
    pool = EDGE + list(range(-20, 21))
    data = {name: [rng.choice(pool) for _ in range(n)] for name in names}
    if names and n >= 5:
        first = names[0]
        data[first][0] = 0
        data[first][1] = 1
        data[first][2] = -1
        data[first][3] = 2147483647
        data[first][4] = -2147483648
    if len(names) > 1 and n >= 5:
        data[names[1]][3] = 0
        data[names[1]][4] = -1
    return data
