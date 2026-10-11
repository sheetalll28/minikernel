from pathlib import Path

import pytest

from minikernel.errors import CheckError, ParseError
from minikernel.frontend import compile_file, compile_source
from minikernel.pretty import kernel_src

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
INVALID = Path(__file__).resolve().parent / "invalid"


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.mk")))
def test_examples_parse(path):
    kernel = compile_file(path)
    again = compile_source(kernel_src(kernel))
    assert again.name == kernel.name
    assert again.inputs == kernel.inputs
    assert again.outputs == kernel.outputs


def test_add_one_roundtrip():
    src = (EXAMPLES / "add_one.mk").read_text()
    kernel = compile_source(src)
    assert "add_one" in kernel_src(kernel)


def test_store_requires_tid():
    with pytest.raises(ParseError, match="only"):
        compile_source("kernel k(out c: int[]) { c[0] = 1; }")


@pytest.mark.parametrize("path", sorted(INVALID.glob("*.mk")))
def test_invalid_rejected(path):
    with pytest.raises((ParseError, CheckError)):
        compile_file(path)
