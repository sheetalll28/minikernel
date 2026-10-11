from pathlib import Path

from .checker import check
from .nodes import Kernel
from .parser import parse


def compile_source(source: str) -> Kernel:
    kernel = parse(source)
    check(kernel)
    return kernel


def compile_file(path: str | Path) -> Kernel:
    return compile_source(Path(path).read_text())
