from minikernel.reduce import ddmin


def test_classic_pair():
    original = [1, 7, 3, 5, 9, 2, 8]
    result = ddmin(original, lambda xs: 7 in xs and 3 in xs)
    assert result == [7, 3]


def test_one_minimal():
    result = ddmin([2, 3, 5], lambda xs: sum(xs) == 10)
    assert sum(result) == 10
    for i in range(len(result)):
        smaller = result[:i] + result[i + 1 :]
        assert sum(smaller) != 10 or not smaller
