from .nodes import Assign, Binary, For, If, Index, Kernel, Let, Num, Store, Tid, Unary, Var

ARITH = {
    "+": "add.s32",
    "-": "sub.s32",
    "*": "mul.lo.s32",
    "&": "and.b32",
    "|": "or.b32",
    "^": "xor.b32",
}
COMPARE = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}


class PTXGenerator:
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.body: list[str] = []
        self.counts = {"r": 0, "rd": 0, "p": 0}
        self.label_count = 0
        self.scopes: list[dict[str, str]] = [{}]
        self.arrays: dict[str, str] = {}
        self.n = ""
        self.tid = ""

    def reg(self, kind="r") -> str:
        self.counts[kind] += 1
        return f"%{kind}{self.counts[kind]}"

    def new_label(self, hint: str) -> str:
        self.label_count += 1
        return f"L_{hint}_{self.label_count}"

    def emit(self, instruction: str) -> None:
        self.body.append(f"    {instruction}")

    def place(self, label: str) -> None:
        self.body.append(f"{label}:")

    def lookup(self, name: str) -> str:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        raise KeyError(f"undeclared variable {name}")

    def generate(self) -> str:
        k = self.kernel
        for name in k.inputs + k.outputs:
            raw, base = self.reg("rd"), self.reg("rd")
            self.emit(f"ld.param.u64 {raw}, [{name}_ptr];")
            self.emit(f"cvta.to.global.u64 {base}, {raw};")
            self.arrays[name] = base

        self.n = self.reg()
        self.emit(f"ld.param.u32 {self.n}, [n_param];")

        block, size, thread = self.reg(), self.reg(), self.reg()
        self.emit(f"mov.u32 {block}, %ctaid.x;")
        self.emit(f"mov.u32 {size}, %ntid.x;")
        self.emit(f"mov.u32 {thread}, %tid.x;")
        self.tid = self.reg()
        self.emit(f"mad.lo.s32 {self.tid}, {block}, {size}, {thread};")

        out_of_range = self.reg("p")
        self.emit(f"setp.ge.u32 {out_of_range}, {self.tid}, {self.n};")
        self.emit(f"@{out_of_range} bra L_done;")

        self.block(k.body)
        self.place("L_done")
        self.emit("ret;")
        return self.assemble()

    def assemble(self) -> str:
        k = self.kernel
        params = [f"    .param .u64 {name}_ptr," for name in k.inputs + k.outputs]
        params.append("    .param .u32 n_param")
        lines = [
            ".version 7.0",
            ".target sm_75",
            ".address_size 64",
            "",
            f".visible .entry {k.name}(",
            *params,
            ")",
            "{",
            f"    .reg .pred %p<{self.counts['p'] + 1}>;",
            f"    .reg .b32 %r<{self.counts['r'] + 1}>;",
            f"    .reg .b64 %rd<{self.counts['rd'] + 1}>;",
            "",
            *self.body,
            "}",
        ]
        return "\n".join(lines) + "\n"

    def block(self, statements):
        self.scopes.append({})
        for stmt in statements:
            self.stmt(stmt)
        self.scopes.pop()

    def stmt(self, node):
        if isinstance(node, Let):
            value = self.expr(node.value)
            var = self.reg()
            self.emit(f"mov.b32 {var}, {value};")
            self.scopes[-1][node.name] = var
        elif isinstance(node, Assign):
            value = self.expr(node.value)
            self.emit(f"mov.b32 {self.lookup(node.name)}, {value};")
        elif isinstance(node, Store):
            value = self.expr(node.value)
            offset, addr = self.reg("rd"), self.reg("rd")
            self.emit(f"mul.wide.u32 {offset}, {self.tid}, 4;")
            self.emit(f"add.s64 {addr}, {self.arrays[node.array]}, {offset};")
            self.emit(f"st.global.u32 [{addr}], {value};")
        elif isinstance(node, If):
            else_label, end_label = self.new_label("else"), self.new_label("endif")
            cond = self.expr(node.cond)
            is_true = self.reg("p")
            self.emit(f"setp.ne.s32 {is_true}, {cond}, 0;")
            self.emit(f"@!{is_true} bra {else_label};")
            self.block(node.then_body)
            self.emit(f"bra.uni {end_label};")
            self.place(else_label)
            self.block(node.else_body)
            self.place(end_label)
        elif isinstance(node, For):
            top, end = self.new_label("loop"), self.new_label("endloop")
            counter = self.reg()
            self.emit(f"mov.s32 {counter}, {node.start};")
            self.place(top)
            finished = self.reg("p")
            self.emit(f"setp.ge.s32 {finished}, {counter}, {node.end};")
            self.emit(f"@{finished} bra {end};")
            self.scopes.append({node.var: counter})
            self.block(node.body)
            self.scopes.pop()
            self.emit(f"add.s32 {counter}, {counter}, 1;")
            self.emit(f"bra.uni {top};")
            self.place(end)
        else:
            raise TypeError(f"not a statement: {node!r}")

    def expr(self, node):
        if isinstance(node, Num):
            r = self.reg()
            self.emit(f"mov.s32 {r}, {node.value};")
            return r
        if isinstance(node, Var):
            return self.lookup(node.name)
        if isinstance(node, Tid):
            return self.tid
        if isinstance(node, Index):
            return self.load(node)
        if isinstance(node, Unary):
            return self.unary(node.op, self.expr(node.operand))
        if isinstance(node, Binary):
            return self.binary(node.op, self.expr(node.left), self.expr(node.right))
        raise TypeError(f"not an expression: {node!r}")

    def load(self, node):
        index = self.expr(node.index)
        in_range = self.reg("p")
        self.emit(f"setp.lt.u32 {in_range}, {index}, {self.n};")
        offset, addr = self.reg("rd"), self.reg("rd")
        self.emit(f"mul.wide.s32 {offset}, {index}, 4;")
        self.emit(f"add.s64 {addr}, {self.arrays[node.array]}, {offset};")
        value = self.reg()
        self.emit(f"mov.u32 {value}, 0;")
        self.emit(f"@{in_range} ld.global.u32 {value}, [{addr}];")
        return value

    def unary(self, op, x):
        r = self.reg()
        if op == "-":
            self.emit(f"neg.s32 {r}, {x};")
        elif op == "~":
            self.emit(f"not.b32 {r}, {x};")
        elif op == "!":
            is_zero = self.reg("p")
            self.emit(f"setp.eq.s32 {is_zero}, {x}, 0;")
            self.emit(f"selp.s32 {r}, 1, 0, {is_zero};")
        else:
            raise ValueError(f"unknown unary operator {op}")
        return r

    def binary(self, op, a, b):
        if op in ("/", "%"):
            return self.div_rem(op, a, b)
        r = self.reg()
        if op in ARITH:
            self.emit(f"{ARITH[op]} {r}, {a}, {b};")
        elif op in COMPARE:
            p = self.reg("p")
            self.emit(f"setp.{COMPARE[op]}.s32 {p}, {a}, {b};")
            self.emit(f"selp.s32 {r}, 1, 0, {p};")
        elif op in ("&&", "||"):
            pa, pb, both = self.reg("p"), self.reg("p"), self.reg("p")
            self.emit(f"setp.ne.s32 {pa}, {a}, 0;")
            self.emit(f"setp.ne.s32 {pb}, {b}, 0;")
            self.emit(f"{'and' if op == '&&' else 'or'}.pred {both}, {pa}, {pb};")
            self.emit(f"selp.s32 {r}, 1, 0, {both};")
        elif op in ("<<", ">>"):
            amount = self.reg()
            self.emit(f"and.b32 {amount}, {b}, 31;")
            self.emit(f"{'shl.b32' if op == '<<' else 'shr.s32'} {r}, {a}, {amount};")
        else:
            raise ValueError(f"unknown binary operator {op}")
        return r

    def div_rem(self, op, a, b):
        is_zero, is_min, is_neg1, overflow, unsafe = (self.reg("p") for _ in range(5))
        self.emit(f"setp.eq.s32 {is_zero}, {b}, 0;")
        self.emit(f"setp.eq.s32 {is_min}, {a}, -2147483648;")
        self.emit(f"setp.eq.s32 {is_neg1}, {b}, -1;")
        self.emit(f"and.pred {overflow}, {is_min}, {is_neg1};")
        self.emit(f"or.pred {unsafe}, {is_zero}, {overflow};")
        safe_b = self.reg()
        self.emit(f"selp.s32 {safe_b}, 1, {b}, {unsafe};")
        raw = self.reg()
        self.emit(f"{'div.s32' if op == '/' else 'rem.s32'} {raw}, {a}, {safe_b};")
        result = self.reg()
        zero_answer = "0" if op == "/" else a
        self.emit(f"selp.s32 {result}, {zero_answer}, {raw}, {is_zero};")
        return result


def compile_to_ptx(kernel: Kernel) -> str:
    return PTXGenerator(kernel).generate()
