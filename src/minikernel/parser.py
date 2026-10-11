from .errors import ParseError
from .lexer import tokenize
from .nodes import Assign, Binary, For, If, Index, Kernel, Let, Num, Store, Tid, Unary, Var


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos]

    def advance(self):
        tok = self.tokens[self.pos]
        if tok.kind != "EOF":
            self.pos += 1
        return tok

    def check(self, text):
        tok = self.peek()
        return tok.kind in ("OP", "KEYWORD") and tok.text == text

    def match(self, *texts):
        for text in texts:
            if self.check(text):
                return self.advance()
        return None

    def error(self, expected):
        tok = self.peek()
        found = tok.text if tok.kind != "EOF" else "end of file"
        return ParseError(f"line {tok.line}: expected {expected} but found {found!r}")

    def expect(self, text):
        if not self.check(text):
            raise self.error(repr(text))
        return self.advance()

    def expect_ident(self):
        if self.peek().kind != "IDENT":
            raise self.error("a name")
        return self.advance()

    def parse_kernel(self):
        self.expect("kernel")
        name = self.expect_ident()
        self.expect("(")
        inputs = []
        outputs = []
        if not self.check(")"):
            while True:
                kind, pname = self.param()
                if kind == "in":
                    inputs.append(pname)
                else:
                    outputs.append(pname)
                if not self.match(","):
                    break
        self.expect(")")
        body = self.block()
        if self.peek().kind != "EOF":
            raise self.error("end of file")
        if not outputs:
            raise ParseError(f"line {name.line}: kernel must declare at least one out array")
        return Kernel(name.text, inputs, outputs, body, name.line)

    def param(self):
        if self.match("in"):
            kind = "in"
        elif self.match("out"):
            kind = "out"
        else:
            raise self.error("'in' or 'out'")
        name = self.expect_ident()
        self.expect(":")
        self.expect("int")
        self.expect("[")
        self.expect("]")
        return kind, name.text

    def block(self):
        self.expect("{")
        statements = []
        while not self.check("}") and self.peek().kind != "EOF":
            statements.append(self.statement())
        self.expect("}")
        return statements

    def statement(self):
        if self.match("let"):
            name = self.expect_ident()
            self.expect("=")
            value = self.expr()
            self.expect(";")
            return Let(name.text, value, name.line)
        if self.match("if"):
            self.expect("(")
            cond = self.expr()
            self.expect(")")
            then_body = self.block()
            else_body = self.block() if self.match("else") else []
            return If(cond, then_body, else_body)
        if self.match("for"):
            var = self.expect_ident()
            self.expect("in")
            start = self.const_int()
            self.expect("..")
            end = self.const_int()
            return For(var.text, start, end, self.block(), var.line)
        if self.peek().kind == "IDENT":
            name = self.advance()
            if self.match("["):
                index = self.expr()
                self.expect("]")
                self.expect("=")
                value = self.expr()
                self.expect(";")
                if not isinstance(index, Tid):
                    raise ParseError(
                        f"line {name.line}: only {name.text}[tid] = ... stores are allowed"
                    )
                return Store(name.text, value, name.line)
            self.expect("=")
            value = self.expr()
            self.expect(";")
            return Assign(name.text, value, name.line)
        raise self.error("a statement")

    def const_int(self):
        sign = -1 if self.match("-") else 1
        tok = self.peek()
        if tok.kind != "NUMBER":
            raise self.error("an integer constant")
        self.advance()
        return sign * int(tok.text)

    def expr(self):
        return self.or_expr()

    def or_expr(self):
        return self.binary(self.and_expr, "||")

    def and_expr(self):
        return self.binary(self.equality, "&&")

    def equality(self):
        return self.binary(self.comparison, "==", "!=")

    def comparison(self):
        return self.binary(self.bit_or, "<", "<=", ">", ">=")

    def bit_or(self):
        return self.binary(self.bit_xor, "|")

    def bit_xor(self):
        return self.binary(self.bit_and, "^")

    def bit_and(self):
        return self.binary(self.shift, "&")

    def shift(self):
        return self.binary(self.term, "<<", ">>")

    def term(self):
        return self.binary(self.factor, "+", "-")

    def factor(self):
        return self.binary(self.unary, "*", "/", "%")

    def binary(self, operand, *ops):
        left = operand()
        while True:
            tok = self.match(*ops)
            if tok is None:
                return left
            left = Binary(tok.text, left, operand())

    def unary(self):
        tok = self.match("-", "!", "~")
        if tok:
            return Unary(tok.text, self.unary())
        return self.primary()

    def primary(self):
        tok = self.peek()
        if tok.kind == "NUMBER":
            self.advance()
            return Num(int(tok.text))
        if self.match("tid"):
            if self.match("["):
                raise ParseError(f"line {tok.line}: tid is not an array")
            return Tid()
        if tok.kind == "IDENT":
            name = self.advance()
            if self.match("["):
                index = self.expr()
                self.expect("]")
                return Index(name.text, index, name.line)
            return Var(name.text, name.line)
        if self.match("("):
            inner = self.expr()
            self.expect(")")
            return inner
        raise self.error("an expression")


def parse(source: str) -> Kernel:
    return Parser(tokenize(source)).parse_kernel()
