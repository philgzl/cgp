import numpy as np

from . import config as cfg
from . import utils


def cgp(x, y, fs, axis=-1, _discard_last_frame=False):
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    if axis < 0:
        axis = x.ndim + axis

    input_shape = x.shape
    if x.ndim == 1:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        axis += 1

    if axis != x.ndim - 1:
        x = x.swapaxes(axis, -1)
        y = y.swapaxes(axis, -1)

    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])

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
    x = np.einsum("nij,ki->nkj", np.abs(x_stft) ** 2, fb**2) ** 0.5
    y = np.einsum("nij,ki->nkj", np.abs(y_stft) ** 2, fb**2) ** 0.5

    x **= cfg.compression_factor
    y **= cfg.compression_factor

    lpf_fs = cfg.fs / cfg.stft_hop_len
    x = utils.apply_lpf(x, cfg.lpf_fc, lpf_fs)
    y = utils.apply_lpf(y, cfg.lpf_fc, lpf_fs)

    gauss_kernel = utils.gauss_kernel(
        cfg.n_channels, cfg.spec_seg_len, cfg.spec_seg_hop, cfg.gauss_kernel_width
    )
    scores = utils.calc_tf_scores(
        x, y, cfg.spec_seg_len, cfg.spec_seg_hop, gauss_kernel
    )

    env_rms = np.nanmean(x**2, axis=-1, keepdims=True) ** 0.5
    env_ratio_dB = 20 * np.log10(x / env_rms)
    mask = env_ratio_dB >= cfg.env_ratio_thr
    glimpse_thrs = np.interp(
        np.arange(cfg.n_channels), cfg.thr_channels, cfg.glimpse_thrs
    )
    glimpses = scores > glimpse_thrs[:, None]
    output = np.sum(glimpses * mask, axis=(1, 2)) / mask.sum(axis=(1, 2))

    if len(input_shape) == 1:
        axis -= 1
    return output.reshape([n for i, n in enumerate(input_shape) if i != axis])
