import matlab.engine
import numpy as np
import pytest
import soundfile as sf

from cgp import cgp


@pytest.mark.parametrize("fs", [10000])
def test_errors(rng, fs):
    with pytest.raises(ValueError):
        cgp(rng.standard_normal(10000), rng.standard_normal(16000), fs)
    with pytest.raises(TypeError):
        cgp(rng.standard_normal(10000), rng.standard_normal(10000), "foo")
    with pytest.raises(TypeError):
        cgp(rng.standard_normal(10000), rng.standard_normal(10000), fs, axis="foo")
    with pytest.raises(TypeError):
        cgp(
            rng.standard_normal(10000),
            rng.standard_normal(10000),
            fs,
            lengths="foo",
        )
    with pytest.raises(TypeError):
        cgp(
            rng.standard_normal((1, 10000)),
            rng.standard_normal((1, 10000)),
            fs,
            lengths="foo",
        )
    with pytest.raises(TypeError):
        cgp(
            rng.standard_normal(10000),
            rng.standard_normal(10000),
            fs,
            lengths=[1000],
        )
    with pytest.raises(TypeError):
        cgp(
            rng.standard_normal((1, 10000)),
            rng.standard_normal((1, 10000)),
            fs,
            lengths=1000,
        )
    with pytest.raises(ValueError):
        cgp(
            rng.standard_normal((1, 10000)),
            rng.standard_normal((1, 10000)),
            fs,
            lengths=[1000, 1000],
        )
    with pytest.raises(TypeError):
        cgp(
            rng.standard_normal((1, 10000)),
            rng.standard_normal((1, 10000)),
            fs,
            _discard_last_frame="foo",
        )


@pytest.mark.parametrize("n", [10000])
@pytest.mark.parametrize("fs", [10000])
def test_silence(n, fs):
    x = np.zeros(n)
    y = np.zeros(n)
    with pytest.raises(RuntimeError):
        cgp(x, y, fs)


@pytest.mark.parametrize("n", [10000, 16384])
@pytest.mark.parametrize("fs", [10000, 16000, 8000])
@pytest.mark.parametrize(
    "silence_idx",
    [
        [(0, 2000)],
        [(0, 2048)],
        [(2000, 4000)],
        [(2000, 4096)],
        [(2048, 4096)],
        [(4000, -1)],
        [(4096, -1)],
        [(0, 2000), (4000, -1)],
        [(0, 2048), (4096, -1)],
        [(2000, 3000), (7000, 8000)],
        [(2048, 3072), (8192, 9216)],
    ],
)
def test_fake_signal(eng, rng, n, fs, silence_idx):
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    for start, end in silence_idx:
        x[start:end] = 0
    cgp_python = cgp(x, y, fs, _discard_last_frame=True)
    cgp_matlab = eng.cgp(
        matlab.double(x.tolist()), matlab.double(y.tolist()), float(fs)
    )
    assert np.allclose(cgp_python, cgp_matlab)


@pytest.mark.parametrize("file_idx", list(range(3)))
def test_real_signal(eng, file_idx):
    x, fs = sf.read(f"tests/audio/{file_idx:05d}_mixture.flac")
    y, fs = sf.read(f"tests/audio/{file_idx:05d}_foreground.flac")
    cgp_python = cgp(x, y, fs, _discard_last_frame=True)
    cgp_matlab = eng.cgp(
        matlab.double(x.tolist()), matlab.double(y.tolist()), float(fs)
    )
    assert np.allclose(cgp_python, cgp_matlab)


@pytest.mark.parametrize("n", [10000, 16384])
@pytest.mark.parametrize("fs", [10000, 16000])
@pytest.mark.parametrize(
    "length", [8190, 8191, 8192, 8193, 8194, 8318, 8319, 8320, 8321, 8322]
)
@pytest.mark.parametrize("_discard_last_frame", [True, False])
def test_length_arg(rng, n, fs, length, _discard_last_frame):
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    cgp_python_1 = cgp(
        x, y, fs, lengths=length, _discard_last_frame=_discard_last_frame
    )
    cgp_python_2 = cgp(
        x[:length], y[:length], fs, _discard_last_frame=_discard_last_frame
    )
    assert np.allclose(cgp_python_1, cgp_python_2)
