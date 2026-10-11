from minikernel.harness import fuzz


def test_small_fuzz_batch():
    results = fuzz(count=3, seed=0, n=8, depth=1)
    kinds = {r.kind for r in results}
    assert "MISMATCH" not in kinds
    assert "TOOL ERROR" not in kinds
    assert results
