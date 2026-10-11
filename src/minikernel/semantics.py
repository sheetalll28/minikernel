"""MiniKernel integer rules. This file is the executable spec (SPEC §5)."""

INT_MIN = -2**31
INT_MAX = 2**31 - 1


def wrap32(x: int) -> int:
    return ((x + 2**31) % 2**32) - 2**31


def mk_div(a: int, b: int) -> int:
    if b == 0:
        return 0
    if a == INT_MIN and b == -1:
        return INT_MIN
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def mk_mod(a: int, b: int) -> int:
    if b == 0:
        return a
    if a == INT_MIN and b == -1:
        return 0
    return a - b * mk_div(a, b)


def mk_shl(a: int, b: int) -> int:
    return wrap32(a << (b & 31))


def mk_shr(a: int, b: int) -> int:
    return a >> (b & 31)


BINARY = {
    "+": lambda a, b: wrap32(a + b),
    "-": lambda a, b: wrap32(a - b),
    "*": lambda a, b: wrap32(a * b),
    "/": mk_div,
    "%": mk_mod,
    "&": lambda a, b: a & b,
    "|": lambda a, b: a | b,
    "^": lambda a, b: a ^ b,
    "<<": mk_shl,
    ">>": mk_shr,
    "==": lambda a, b: int(a == b),
    "!=": lambda a, b: int(a != b),
    "<": lambda a, b: int(a < b),
    "<=": lambda a, b: int(a <= b),
    ">": lambda a, b: int(a > b),
    ">=": lambda a, b: int(a >= b),
    "&&": lambda a, b: int(a != 0 and b != 0),
    "||": lambda a, b: int(a != 0 or b != 0),
}

UNARY = {
    "-": lambda a: wrap32(-a),
    "!": lambda a: int(a == 0),
    "~": lambda a: ~a,
}

# Matches the parser: lower binds looser. Used only for pretty-printing.
PRECEDENCE = {
    "||": 1,
    "&&": 2,
    "==": 3,
    "!=": 3,
    "<": 4,
    "<=": 4,
    ">": 4,
    ">=": 4,
    "|": 5,
    "^": 6,
    "&": 7,
    "<<": 8,
    ">>": 8,
    "+": 9,
    "-": 9,
    "*": 10,
    "/": 10,
    "%": 10,
}
