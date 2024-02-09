import numpy as np

from . import config as cfg
from . import utils


def cgp(x, y, fs, _discard_last_frame=False):
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    if fs != cfg.fs:
        x = utils.resample(x, fs, cfg.fs)
        y = utils.resample(y, fs, cfg.fs)

    x, y = utils.remove_silent_frames(
        x,
        y,
        cfg.vad_win_len,
        cfg.vad_hop_len,
        cfg.vad_dyn_range,
        _discard_last_frame=_discard_last_frame,
    )

    x_stft = utils.stft(x, cfg.stft_win_len, cfg.stft_hop_len, cfg.stft_n_fft)
    y_stft = utils.stft(y, cfg.stft_win_len, cfg.stft_hop_len, cfg.stft_n_fft)

    fb = utils.gen_cochlear_fb(cfg.fs, cfg.stft_n_fft, cfg.fcs)
    x = np.dot(fb**2, np.abs(x_stft) ** 2) ** 0.5
    y = np.dot(fb**2, np.abs(y_stft) ** 2) ** 0.5

    x **= cfg.compression_factor
    y **= cfg.compression_factor

    lpf_n_fft = 2 ** np.ceil(np.log2(x.shape[-1])).astype(int)
    lpf_fs = cfg.fs / cfg.stft_hop_len
    lpf = utils.gen_tmp_mod_lpf(cfg.lpf_fc, lpf_fs, lpf_n_fft)
    x = utils.apply_lpf(x, lpf)
    y = utils.apply_lpf(y, lpf)

    gauss_kernel = utils.gauss_kernel(
        cfg.n_channels, cfg.spec_seg_len, cfg.spec_seg_hop, cfg.gauss_kernel_width
    )
    scores = utils.calc_tf_scores(
        x, y, cfg.spec_seg_len, cfg.spec_seg_hop, gauss_kernel
    )

    env_rms = (x**2).mean(axis=1, keepdims=True) ** 0.5
    env_ratio_dB = 20 * np.log10(x / env_rms)
    glimpse_thrs = np.interp(
        np.arange(cfg.n_channels), cfg.thr_channels, cfg.glimpse_thrs
    )
    glimpses = scores > glimpse_thrs[:, None]
    return glimpses[env_ratio_dB >= cfg.env_ratio_thr].mean()
