from pyquil.job_results import _round_to_next_multiple, _octet_bits


def test_rounding():
    for i in range(8):
        if 0 == i % 8:
            assert i == _round_to_next_multiple(i, 8)
        else:
            assert 8 == _round_to_next_multiple(i, 8)
            assert 16 == _round_to_next_multiple(i + 8, 8)
            assert 24 == _round_to_next_multiple(i + 16, 8)


def test_octet_bits():
    assert [0, 0, 0, 0, 0, 0, 0, 0] == _octet_bits(0b0)
    assert [1, 0, 0, 0, 0, 0, 0, 0] == _octet_bits(0b1)
    assert [0, 1, 0, 0, 0, 0, 0, 0] == _octet_bits(0b10)
    assert [1, 0, 1, 0, 0, 0, 0, 0] == _octet_bits(0b101)
    assert [1, 1, 1, 1, 1, 1, 1, 1] == _octet_bits(0b11111111)
