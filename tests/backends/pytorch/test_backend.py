from geometric_kernels import BACKEND


def test_backend():
    assert BACKEND == "pytorch"
