import numpy as np

from . import config as cfg
from . import utils


def cgp(x, y, fs, axis=-1, lengths=None, _discard_last_frame=False):
    """Correlation based glimpse proportion (CGP) index.

    Proposed in [1]_.

    Parameters
    ----------
    x : ndarray
        Input signal. Can be multi-dimensional.
    y : ndarray
        Reference signal. Same shape as `x`.
    fs : int
        Sampling frequency.
    axis : int, optional
        The axis along which the CGP is calculated. Default is `-1`.
    lengths : list or ndarray, optional
        Lengths of input signals. Useful for batched inputs if zero-padding is used.
        Default is `None`, i.e. the full-length inputs are used.
    _discard_last_frame : bool, optional
        Whether to discard the last VAD frame as in the original MATLAB implementation.
        Default is `False`.

    Returns
    -------
    ndarray
        CGP values.

    References
    ----------
    .. [1] A. Alghamdi, L. Moen, W.-Y. Chan, D. Fogerty and J. Jensen, "Correlation
           Based Glimpse Proportion Index", in Proc. WASPAA, 2023.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError(
            f"x and y must be numpy arrays, got {x.__class__.__name__} and "
            f"{y.__class__.__name__}"
        )

    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape, got {x.shape} and {y.shape}"
        )

    if not isinstance(fs, (int, np.integer)):
        raise TypeError(f"fs must be an integer, got {fs.__class__.__name__}")

    if not isinstance(axis, (int, np.integer)):
        raise TypeError(f"axis must be an integer, got {axis.__class__.__name__}")

    if not isinstance(_discard_last_frame, bool):
        raise TypeError(
            "_discard_last_frame must be a boolean, got "
            f"{type(_discard_last_frame).__name__}"
        )

    if axis < 0:
        axis = x.ndim + axis

    input_shape = x.shape
    if x.ndim == 1:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        axis += 1
        if lengths is None:
            lengths = [x.shape[-1]]
        elif not isinstance(lengths, (int, np.integer)):
            raise TypeError(
                "lengths must be an integer for one-dimensional inputs, got "
                f"{lengths.__class__.__name__}"
            )
        else:
            lengths = [lengths]
    elif lengths is not None and not isinstance(lengths, (list, np.ndarray)):
        raise TypeError(
            "lengths must be a list or an array for multi-dimensional inputs, got "
            f"{lengths.__class__.__name__}"
        )

    if axis != x.ndim - 1:
        x = np.moveaxis(x, axis, -1)
        y = np.moveaxis(y, axis, -1)

    if lengths is not None:
        lengths = np.array(lengths)
        if lengths.shape != x.shape[:-1]:
            raise ValueError(
                "the shape of lengths must be the same as the inputs without the axis "
                f"dimension, got {lengths.shape} and {x.shape[:-1]}"
            )

    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])

    if lengths is None:
        lengths = np.full(x.shape[0], x.shape[-1], dtype=int)
    else:
        lengths = lengths.flatten()

    if fs != cfg.fs:
        x = utils.resample(x, fs, cfg.fs)
        y = utils.resample(y, fs, cfg.fs)
        lengths = np.ceil(lengths * cfg.fs / fs).astype(int)

    for i, length in enumerate(lengths):
        x[i, length - _discard_last_frame :] = np.nan
        y[i, length - _discard_last_frame :] = np.nan

    x, y = utils.remove_silent_frames(
        x,
        y,
        cfg.vad_win_len,
        cfg.vad_hop_len,
        cfg.vad_dyn_range,
        _discard_last_frame=_discard_last_frame,
    )

    x = utils.stft(x, cfg.stft_win_len, cfg.stft_hop_len, cfg.stft_n_fft)
    y = utils.stft(y, cfg.stft_win_len, cfg.stft_hop_len, cfg.stft_n_fft)

    fb = utils.gen_cochlear_fb(cfg.fs, cfg.stft_n_fft, cfg.fcs)
    x = np.einsum("nij,ki->nkj", np.abs(x) ** 2, fb**2) ** 0.5
    y = np.einsum("nij,ki->nkj", np.abs(y) ** 2, fb**2) ** 0.5

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
