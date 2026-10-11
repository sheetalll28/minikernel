from minikernel.semantics import INT_MIN, mk_div, mk_mod, mk_shl, mk_shr, wrap32


def test_div_by_zero():
    assert mk_div(7, 0) == 0
    assert mk_mod(7, 0) == 7


def test_div_min_neg1():
    assert mk_div(INT_MIN, -1) == INT_MIN
    assert mk_mod(INT_MIN, -1) == 0


def test_div_toward_zero():
    assert mk_div(-7, 2) == -3
    assert mk_mod(-7, 2) == -1


def test_identity():
    for a in (7, -7, INT_MIN, 0, 2147483647):
        for b in (2, -2, 0, -1, 1):
            assert wrap32(mk_div(a, b) * b + mk_mod(a, b)) == a


def test_shifts():
    assert mk_shl(1, 33) == 2
    assert mk_shr(-8, 1) == -4
    assert mk_shr(-8, 32) == -8


def test_wrap():
    assert wrap32(2147483647 + 1) == INT_MIN
    assert wrap32(INT_MIN - 1) == 2147483647
