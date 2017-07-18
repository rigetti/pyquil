from pyquil.job_results import _round_to_next_multiple, _octet_bits
from pyquil.job_results import JobResult
from pyquil.qpu import QPUConnection
import mock
from mock import patch


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


def test_get_is_called():
    # check that JobResult.is_done() calls .get() on QPU prior to
    # returning result
    qpu_connect = QPUConnection('test')
    job_result = JobResult(qpu_connect, False)
    try:
        with patch.object(job_result, 'get') as mock:
            job_result.is_done()
    except:
        pass
    mock.assert_called_with()
