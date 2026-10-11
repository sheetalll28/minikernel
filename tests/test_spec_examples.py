from pathlib import Path

from minikernel.frontend import compile_source
from minikernel.interpreter import interpret

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "spec" / "spec_examples.txt"


def test_spec_examples():
    for raw in EXAMPLES.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        expr, expected = [part.strip() for part in line.split("=>", 1)]
        source = (
            "kernel spec(in a: int[], in b: int[], out c: int[]) {\n"
            f"    c[tid] = {expr};\n"
            "}\n"
        )
        kernel = compile_source(source)
        out = interpret(kernel, {"a": [7], "b": [2]}, 1)
        assert out["c"][0] == int(expected), f"{expr}: expected {expected}, got {out['c'][0]}"
