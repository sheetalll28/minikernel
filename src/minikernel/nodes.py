from dataclasses import dataclass, field


@dataclass
class Num:
    value: int


@dataclass
class Var:
    name: str
    line: int = 0


@dataclass
class Tid:
    pass


@dataclass
class Index:
    array: str
    index: object
    line: int = 0


@dataclass
class Unary:
    op: str
    operand: object


@dataclass
class Binary:
    op: str
    left: object
    right: object


@dataclass
class Let:
    name: str
    value: object
    line: int = 0


@dataclass
class Assign:
    name: str
    value: object
    line: int = 0


@dataclass
class Store:
    array: str
    value: object
    line: int = 0


@dataclass
class If:
    cond: object
    then_body: list
    else_body: list


@dataclass
class For:
    var: str
    start: int
    end: int
    body: list
    line: int = 0


@dataclass
class Kernel:
    name: str
    inputs: list[str]
    outputs: list[str]
    body: list
    line: int = 1


def walk(node):
    """Yield every node in a tree, parents before children."""
    yield node
    if isinstance(node, Kernel):
        for stmt in node.body:
            yield from walk(stmt)
    elif isinstance(node, (Let, Assign, Store)):
        yield from walk(node.value)
    elif isinstance(node, If):
        yield from walk(node.cond)
        for stmt in node.then_body + node.else_body:
            yield from walk(stmt)
    elif isinstance(node, For):
        for stmt in node.body:
            yield from walk(stmt)
    elif isinstance(node, Unary):
        yield from walk(node.operand)
    elif isinstance(node, Binary):
        yield from walk(node.left)
        yield from walk(node.right)
    elif isinstance(node, Index):
        yield from walk(node.index)


def copy_kernel(kernel: Kernel, body=None) -> Kernel:
    return Kernel(
        name=kernel.name,
        inputs=list(kernel.inputs),
        outputs=list(kernel.outputs),
        body=kernel.body if body is None else body,
        line=kernel.line,
    )
