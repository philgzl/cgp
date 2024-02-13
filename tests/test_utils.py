import matlab.engine
import numpy as np
import pytest

import cgp.config as cfg
import cgp.utils


@pytest.mark.parametrize("n", [1023, 1024])
def test_hanning(eng, n):
    hanning_python = cgp.utils.hanning(n)[:, None]
    hanning_matlab = eng.hanning(float(n))
    assert hanning_python.shape == hanning_matlab.size
    assert np.allclose(hanning_python, hanning_matlab)


@pytest.mark.parametrize("n", [10000, 16384])
@pytest.mark.parametrize("n_fft", [512, 2048])
@pytest.mark.parametrize("hop_len", [128, 256])
@pytest.mark.parametrize("win_len", [256, 512])
def test_stft(eng, rng, n, n_fft, hop_len, win_len):
    x = rng.standard_normal(n)
    stft_python = cgp.utils.stft(x, win_len=win_len, hop_len=hop_len, n_fft=n_fft)
    stft_matlab = eng.stdft(
        matlab.double(x.tolist()), float(win_len), float(hop_len), float(n_fft)
    )
    assert stft_python.shape == stft_matlab.size
    assert np.allclose(stft_python, stft_matlab)


@pytest.mark.parametrize("n", [10000, 16384])
@pytest.mark.parametrize("fs_in", [10000])
@pytest.mark.parametrize("fs_out", [16000])
def test_resample(eng, rng, n, fs_in, fs_out):
    x = rng.standard_normal(n)
    resampled_python = cgp.utils.resample(x, fs_in, fs_out)
    resampled_matlab = eng.resample(
        matlab.double(x.tolist()), float(fs_out), float(fs_in)
    )
    resampled_python = resampled_python[None, :]
    assert resampled_python.shape == resampled_matlab.size
    assert np.allclose(resampled_python, resampled_matlab)


@pytest.mark.parametrize("n", [10000, 16384])
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
def test_remove_silent_frames(eng, rng, n, silence_idx):
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    for start, end in silence_idx:
        x[start:end] = 0
    x_python, y_python = cgp.utils.remove_silent_frames(
        x[None, :],
        y[None, :],
        cfg.vad_win_len,
        cfg.vad_hop_len,
        cfg.vad_dyn_range,
        _discard_last_frame=True,
    )
    x_matlab, y_matlab = eng.removeSilentFrames(
        matlab.double(x.tolist()),
        matlab.double(y.tolist()),
        float(cfg.vad_dyn_range),
        float(cfg.vad_win_len),
        float(cfg.vad_hop_len),
        nargout=2,
    )
    assert x_python.T.shape == x_matlab.size
    assert y_python.T.shape == y_matlab.size
    assert np.allclose(x_python.T, x_matlab)
    assert np.allclose(y_python.T, y_matlab)


def test_gen_cochlear_fb(eng):
    fb_python = cgp.utils.gen_cochlear_fb(cfg.fs, cfg.stft_n_fft, cfg.fcs)
    fb_matlab = eng.gen_cochlear_fb(
        float(cfg.fs), float(cfg.stft_n_fft), matlab.double(cfg.fcs.tolist())
    )
    assert fb_python.shape == fb_matlab.size
    assert np.allclose(fb_python, fb_matlab)


def test_gen_tmp_mod_lpf(eng):
    n_fft = 2 ** np.ceil(np.log2(10000)).astype(int)
    stft_fs = cfg.fs / cfg.stft_hop_len
    lpf_python = cgp.utils.gen_tmp_mod_lpf(cfg.lpf_fc, stft_fs, n_fft)
    lpf_matlab = eng.gen_tmp_mod_LPF(float(cfg.lpf_fc), float(stft_fs), float(n_fft))
    lpf_python = lpf_python[None, :]
    assert lpf_python.shape == lpf_matlab.size
    assert np.allclose(lpf_python, lpf_matlab)


@pytest.mark.parametrize("n", [10000, 16384])
def test_calc_tf_scores(eng, n):
    x = np.random.standard_normal((cfg.n_channels, n))
    y = np.random.standard_normal((cfg.n_channels, n))
    gauss_kernel = cgp.utils.gauss_kernel(
        cfg.n_channels, cfg.spec_seg_len, cfg.spec_seg_hop, cfg.gauss_kernel_width
    )
    scores_python = cgp.utils.calc_tf_scores(
        x[None, :], y[None, :], cfg.spec_seg_len, cfg.spec_seg_hop, gauss_kernel
    ).squeeze(0)
    scores_matlab = eng.calc_TF_scores(
        matlab.double(x.tolist()),
        matlab.double(y.tolist()),
        float(cfg.spec_seg_len),
        float(cfg.spec_seg_hop),
        matlab.double(gauss_kernel.tolist()),
    )
    assert scores_python.shape == scores_matlab.size
    assert np.allclose(scores_python, scores_matlab)
