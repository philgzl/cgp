import numpy as np
import pytest
import soundfile as sf

from cgp import cgp


@pytest.mark.parametrize(
    "input_shape",
    [(1, 8000), (2, 8000), (4, 8000), (4, 1, 8000), (1, 4, 8000), (16, 16, 8000)],
)
@pytest.mark.parametrize("fs", [10000, 16000])
def test_fake_signal(rng, input_shape, fs):
    x = rng.standard_normal(input_shape)
    y = rng.standard_normal(input_shape)

    # add two random silences in each item
    input_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    for i in range(x.shape[0]):
        j_start_1 = rng.integers(0, x.shape[-1])
        j_end_1 = rng.integers(j_start_1, x.shape[-1])
        j_start_2 = rng.integers(j_end_1, x.shape[-1])
        j_end_2 = rng.integers(j_start_2, x.shape[-1])
        x[i, j_start_1:j_end_1] = 0
        x[i, j_start_2:j_end_2] = 0
    x = x.reshape(input_shape)

    batched_cgp = cgp(x, y, fs).flatten()
    unbatched_cgp = np.array(
        [
            cgp(x_i, y_i, fs)
            for x_i, y_i in zip(x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]))
        ]
    )
    assert np.allclose(batched_cgp, unbatched_cgp)


def test_real_signal():
    xs, ys = [], []
    for file_idx in range(3):
        x, fs = sf.read(f"tests/audio/{file_idx:05d}_mixture.flac")
        y, fs = sf.read(f"tests/audio/{file_idx:05d}_foreground.flac")
        xs.append(x)
        ys.append(y)
    n_max = max(len(x) for x in xs)
    x = np.stack([np.pad(x, (0, n_max - len(x))) for x in xs])
    y = np.stack([np.pad(y, (0, n_max - len(y))) for y in ys])

    batched_cgp = cgp(x, y, fs)
    unbatched_cgp = np.array([cgp(x_i, y_i, fs) for x_i, y_i in zip(x, y)])
    assert np.allclose(batched_cgp, unbatched_cgp)
