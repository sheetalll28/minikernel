from dataclasses import dataclass

from .errors import LexError

KEYWORDS = {
    "kernel",
    "in",
    "out",
    "int",
    "let",
    "if",
    "else",
    "for",
    "tid",
}

TWO_CHAR_OPS = {"==", "!=", "<=", ">=", "&&", "||", "<<", ">>", ".."}
ONE_CHAR_OPS = set("+-*/%<>=!(){};:,[]~^&|")


@dataclass
class Token:
    kind: str
    text: str
    line: int


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    line = 1
    while i < len(source):
        ch = source[i]
        if ch == "\n":
            line += 1
            i += 1
        elif ch.isspace():
            i += 1
        elif source.startswith("//", i):
            while i < len(source) and source[i] != "\n":
                i += 1
        elif ch.isdigit():
            start = i
            while i < len(source) and source[i].isdigit():
                i += 1
            tokens.append(Token("NUMBER", source[start:i], line))
        elif ch.isalpha() or ch == "_":
            start = i
            while i < len(source) and (source[i].isalnum() or source[i] == "_"):
                i += 1
            word = source[start:i]
            kind = "KEYWORD" if word in KEYWORDS else "IDENT"
            tokens.append(Token(kind, word, line))
        elif source[i : i + 2] in TWO_CHAR_OPS:
            tokens.append(Token("OP", source[i : i + 2], line))
            i += 2
        elif ch in ONE_CHAR_OPS:
            tokens.append(Token("OP", ch, line))
            i += 1
        else:
            raise LexError(f"line {line}: unexpected character {ch!r}")
    tokens.append(Token("EOF", "", line))
    return tokens
