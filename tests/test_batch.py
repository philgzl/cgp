import numpy as np
import pytest
import soundfile as sf

from cgp import cgp


@pytest.mark.parametrize(
    "input_shape",
    [(1, 8000), (2, 8000), (4, 8000), (4, 1, 8000), (1, 4, 8000), (16, 12, 8000)],
)
@pytest.mark.parametrize("fs", [10000, 16000, 8000])
@pytest.mark.parametrize("_discard_last_frame", [True, False])
def test_fake_signal(rng, input_shape, fs, _discard_last_frame):
    x_batched = rng.standard_normal(input_shape)
    y_batched = rng.standard_normal(input_shape)

    # add two random silences in each item an set random length
    input_shape = x_batched.shape
    x_batched = x_batched.reshape(-1, x_batched.shape[-1])
    y_batched = y_batched.reshape(-1, y_batched.shape[-1])
    lengths = []
    x_unbatched = []
    y_unbatched = []
    for i in range(x_batched.shape[0]):
        j_start_1 = rng.integers(0, x_batched.shape[-1])
        j_end_1 = rng.integers(j_start_1, x_batched.shape[-1])
        j_start_2 = rng.integers(j_end_1, x_batched.shape[-1])
        j_end_2 = rng.integers(j_start_2, x_batched.shape[-1])
        length = rng.integers(j_end_2, x_batched.shape[-1])
        x_batched[i, j_start_1:j_end_1] = 0
        x_batched[i, j_start_2:j_end_2] = 0
        lengths.append(length)
        x_unbatched.append(x_batched[i, :length])
        y_unbatched.append(y_batched[i, :length])
    x_batched = x_batched.reshape(input_shape)
    y_batched = y_batched.reshape(input_shape)
    lengths = np.array(lengths).reshape(input_shape[:-1])

    batched_cgp = cgp(
        x_batched,
        y_batched,
        fs,
        lengths=lengths,
        _discard_last_frame=_discard_last_frame,
    ).flatten()
    unbatched_cgp = np.array(
        [
            cgp(x, y, fs, _discard_last_frame=_discard_last_frame)
            for x, y in zip(x_unbatched, y_unbatched)
        ]
    )
    assert np.allclose(batched_cgp, unbatched_cgp)


@pytest.mark.parametrize("_discard_last_frame", [True, False])
def test_real_signal(_discard_last_frame):
    x_unbatched, y_unbatched = [], []
    for file_idx in range(3):
        x, fs = sf.read(f"tests/audio/{file_idx:05d}_mixture.flac")
        y, fs = sf.read(f"tests/audio/{file_idx:05d}_foreground.flac")
        x_unbatched.append(x)
        y_unbatched.append(y)
    lengths = [len(x) for x in x_unbatched]
    n_max = max(lengths)
    x_batched = np.stack([np.pad(x, (0, n_max - len(x))) for x in x_unbatched])
    y_batched = np.stack([np.pad(y, (0, n_max - len(y))) for y in y_unbatched])

    batched_cgp = cgp(
        x_batched,
        y_batched,
        fs,
        lengths=lengths,
        _discard_last_frame=_discard_last_frame,
    )
    unbatched_cgp = np.array(
        [
            cgp(x, y, fs, _discard_last_frame=_discard_last_frame)
            for x, y in zip(x_unbatched, y_unbatched)
        ]
    )
    assert np.allclose(batched_cgp, unbatched_cgp)


@pytest.mark.parametrize(
    "input_shape",
    [(2, 8000), (2, 4, 8, 8000)],
)
@pytest.mark.parametrize("fs", [10000, 16000])
def test_axis(rng, input_shape, fs):
    x = rng.standard_normal(input_shape)
    y = rng.standard_normal(input_shape)

    cgp_0 = cgp(x, y, fs, axis=-1)
    assert cgp_0.shape == x.shape[:-1]

    for i in range(x.ndim):
        cgp_i = cgp(np.moveaxis(x, -1, i), np.moveaxis(y, -1, i), fs, axis=i)

        assert np.allclose(cgp_0, cgp_i)
