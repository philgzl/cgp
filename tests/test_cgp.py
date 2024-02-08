import matlab.engine
import numpy as np
import pytest

from cgp import cgp


@pytest.mark.parametrize('n', [10000])
@pytest.mark.parametrize('fs', [10000])
def test_silence(n, fs):
    # The MATLAB implementation raises an error for all-zero inputs.
    # Maybe this behavior should be changed in the future.
    x = np.zeros(n)
    y = np.zeros(n)
    with pytest.raises(RuntimeError):
        cgp(x, y, fs)


@pytest.mark.parametrize('n', [10000, 16384])
@pytest.mark.parametrize('fs', [10000, 16000])
@pytest.mark.parametrize('silence_idx', [
    [(0, 2000), (4000, -1)],
    [(0, 2048), (4096, -1)],
    [(2000, 3000), (7000, 8000)],
    [(2048, 3072), (8192, 9216)]
])
def test_cgp(eng, n, fs, silence_idx):
    rng = np.random.default_rng(seed=0)
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    for start, end in silence_idx:
        x[start:end] = 0
    cgp_python = cgp(x, y, fs)
    cgp_matlab = eng.cgp(
        matlab.double(x.tolist()),
        matlab.double(y.tolist()),
        float(fs)
    )
    assert np.allclose(cgp_python, cgp_matlab)
