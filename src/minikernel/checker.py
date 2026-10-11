from .errors import CheckError
from .nodes import Assign, Binary, For, If, Index, Kernel, Let, Store, Unary, Var


class Checker:
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.errors: list[str] = []
        self.loop_vars: list[str] = []
        self.scopes: list[set[str]] = [set()]

    def is_declared(self, name: str) -> bool:
        return any(name in scope for scope in self.scopes)

    def check(self) -> list[str]:
        seen = set()
        for name in self.kernel.inputs + self.kernel.outputs:
            if name in seen:
                self.errors.append(f"line {self.kernel.line}: duplicate parameter '{name}'")
            seen.add(name)
        overlap = set(self.kernel.inputs) & set(self.kernel.outputs)
        for name in sorted(overlap):
            self.errors.append(f"line {self.kernel.line}: '{name}' cannot be both in and out")
        self.check_block(self.kernel.body)
        return self.errors

    def check_block(self, body):
        self.scopes.append(set())
        for stmt in body:
            self.check_stmt(stmt)
        self.scopes.pop()

    def check_stmt(self, stmt):
        if isinstance(stmt, Let):
            self.check_expr(stmt.value)
            if stmt.name in self.scopes[-1]:
                self.errors.append(f"line {stmt.line}: '{stmt.name}' is already declared in this block")
            if stmt.name in self.kernel.inputs or stmt.name in self.kernel.outputs:
                self.errors.append(f"line {stmt.line}: '{stmt.name}' shadows a kernel parameter")
            self.scopes[-1].add(stmt.name)
        elif isinstance(stmt, Assign):
            self.check_expr(stmt.value)
            if stmt.name in self.loop_vars:
                self.errors.append(f"line {stmt.line}: cannot assign to loop variable '{stmt.name}'")
            elif not self.is_declared(stmt.name):
                self.errors.append(f"line {stmt.line}: assignment to undeclared variable '{stmt.name}'")
        elif isinstance(stmt, Store):
            self.check_expr(stmt.value)
            if stmt.array not in self.kernel.outputs:
                self.errors.append(f"line {stmt.line}: can only write an out array, not '{stmt.array}'")
        elif isinstance(stmt, If):
            self.check_expr(stmt.cond)
            self.check_block(stmt.then_body)
            self.check_block(stmt.else_body)
        elif isinstance(stmt, For):
            if stmt.var in self.scopes[-1] or stmt.var in self.loop_vars:
                self.errors.append(f"line {stmt.line}: loop variable '{stmt.var}' is already in use")
            self.loop_vars.append(stmt.var)
            self.scopes.append({stmt.var})
            self.check_block(stmt.body)
            self.scopes.pop()
            self.loop_vars.pop()

    def check_expr(self, node):
        if isinstance(node, Var):
            if not self.is_declared(node.name):
                self.errors.append(f"line {node.line}: variable '{node.name}' used before 'let'")
        elif isinstance(node, Index):
            if node.array not in self.kernel.inputs:
                self.errors.append(f"line {node.line}: can only read an in array, not '{node.array}'")
            self.check_expr(node.index)
        elif isinstance(node, Unary):
            self.check_expr(node.operand)
        elif isinstance(node, Binary):
            self.check_expr(node.left)
            self.check_expr(node.right)


def check(kernel: Kernel) -> None:
    errors = Checker(kernel).check()
    if errors:
        raise CheckError("\n".join(errors))
