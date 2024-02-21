import numpy as np
import scipy.signal


def unfold(x, win_len, hop_len, axis=-1):
    n_frames = (x.shape[axis] - win_len) // hop_len + 1
    idx = np.add.outer(np.arange(win_len), np.arange(n_frames) * hop_len)
    return np.take(x, idx, axis=axis)


def hanning(n):
    win = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, n + 1) / (n + 1)))
    return win


def stft(x, win_len, hop_len, n_fft):
    x = unfold(x, win_len, hop_len, axis=-1) * hanning(win_len)[:, None]
    return np.fft.rfft(x, n=n_fft, axis=-2)


def resample(x, fs_in, fs_out):
    return scipy.signal.resample_poly(x, fs_out, fs_in, axis=-1)


def remove_silent_frames(x, y, win_len, hop_len, dyn_range, _discard_last_frame=False):
    # Inspired from https://github.com/mpariente/pystoi
    # Copyright (c) 2018 Pariente Manuel
    # MIT License
    x_frames = unfold(x, win_len, hop_len) * hanning(win_len)[:, None]
    y_frames = unfold(y, win_len, hop_len) * hanning(win_len)[:, None]

    # The original MATLAB implementation discards the last frame if the signal fits an
    # integer number of frames
    if _discard_last_frame and (x.shape[-1] - win_len) % hop_len == 0:
        x_frames = x_frames[..., :-1]
        y_frames = y_frames[..., :-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        x_dB = 20 * np.log10(np.linalg.norm(x_frames, axis=-2))
        mask = (np.nanmax(x_dB, axis=-1, keepdims=True) - dyn_range - x_dB) < 0

    if not all(mask.sum(axis=-1) > 2):
        (idx,) = np.where(mask.sum(axis=-1) <= 2)
        raise RuntimeError(
            "Less than 3 speech frames were detected in batch items with index "
            f"{idx.tolist()}"
        )

    x_frames = apply_mask(x_frames, mask)
    y_frames = apply_mask(y_frames, mask)

    x = overlap_and_add(x_frames, hop_len)
    y = overlap_and_add(y_frames, hop_len)

    return x, y


def apply_mask(x, mask):
    ns = mask.sum(axis=-1)
    n_max = ns.max()
    return np.stack(
        [
            np.pad(
                x[i, :, mask[i]], ((0, n_max - ns[i]), (0, 0)), constant_values=np.nan
            )
            for i in range(x.shape[0])
        ]
    )


def overlap_and_add(x, hop_len):
    # Inspired from https://github.com/mpariente/pystoi
    # Copyright (c) 2018 Pariente Manuel
    # MIT License
    # Original PR by Gianluca Micchi
    # https://github.com/mpariente/pystoi/pull/26
    batch_size, n_frames, win_len = x.shape

    # Compute the number of segments per frame
    n_seg = -(-win_len // hop_len)  # Divide and round up

    # Get non-nan frames
    x = [x_i[~np.isnan(x_i).any(axis=1), :] for x_i in x]
    n_valid_frames = [x_i.shape[0] for x_i in x]

    # Pad the win_len dimension to n_seg * hop_len and add n_seg frames
    x = [np.pad(x_i, ((0, n_seg), (0, n_seg * hop_len - win_len))) for x_i in x]

    # Restore nans
    x = np.stack(
        [
            np.pad(x_i, ((0, n_frames - n_i), (0, 0)), constant_values=np.nan)
            for x_i, n_i in zip(x, n_valid_frames)
        ]
    )

    # Reshape to a 4D tensor, splitting the win_len dimension in two
    x = x.reshape(batch_size, n_frames + n_seg, n_seg, hop_len)

    # Swap axes so that x.shape = (batch_size, n_seg, n_frames + n_seg, hop_len)
    x = x.swapaxes(1, 2)

    # Reshape so that x.shape = (batch_size, n_seg * (n_frames + n_seg), hop_len)
    x = x.reshape(batch_size, -1, hop_len)

    # Now behold the magic! Remove the last n_seg elements from the second axis
    x = x[:, :-n_seg]

    # Reshape to (batch_size, n_seg, n_frames + n_seg - 1, hop_len)
    x = x.reshape((batch_size, n_seg, n_frames + n_seg - 1, hop_len))
    # This has introduced a shift by one in all rows

    # Now reduce over the columns and flatten the array to achieve the result
    x = np.nansum(x, axis=1).reshape(batch_size, -1)
    for i in range(batch_size):
        end = (n_valid_frames[i] - 1) * hop_len + win_len
        x[i, end:] = np.nan
    return x


def gen_cochlear_fb(fs, n_fft, fcs):
    erb = 24.7 + fcs / 9.26449
    i = np.arange(n_fft // 2 + 1)
    f = i * fs / n_fft
    fl, fr = fcs - erb / 2, fcs + erb / 2
    il = np.abs(np.subtract.outer(f, fl)).argmin(axis=0)
    ir = np.abs(np.subtract.outer(f, fr)).argmin(axis=0)
    return np.less_equal.outer(il, i) & np.greater.outer(ir, i)


def gen_tmp_mod_lpf(fc, fs, n_fft):
    f = np.arange(n_fft) * fs / n_fft
    f[f > fs / 2] -= fs
    mask = np.abs(f) <= fc
    H = mask.astype(float)
    f_over_fc = f[~mask] / fc
    H[~mask] = f_over_fc**2 * np.exp(1 - f_over_fc**2)
    return H / np.max(H)


def apply_lpf(x, fc, fs):
    # TODO: find a more efficient method for batched inputs
    if x.ndim == 2:
        n_orig = x.shape[-1]
        x = x[:, ~np.isnan(x).any(axis=0)]
        n_fft = 2 ** np.ceil(np.log2(x.shape[-1])).astype(int)
        lpf = gen_tmp_mod_lpf(fc, fs, n_fft)
        x_fft = np.fft.fft(x, len(lpf), axis=-1)
        x = np.fft.ifft(lpf * x_fft, axis=-1).real[..., : x.shape[-1]]
        x = np.pad(x, ((0, 0), (0, n_orig - x.shape[-1])), constant_values=np.nan)
    else:
        x = np.stack([apply_lpf(x[i], fc, fs) for i in range(x.shape[0])])
    return x


def gauss_kernel(n_channels, spec_seg_len, spec_seg_hop, width):
    seg_idx = np.arange(0, n_channels - spec_seg_len + spec_seg_hop + 1, spec_seg_hop)
    j = np.arange(n_channels)
    gauss_kernel = np.exp(-0.5 * (np.subtract.outer(j, seg_idx) / width) ** 2)
    return gauss_kernel / gauss_kernel.sum(axis=1, keepdims=True)


def normalize(x, axis, mean_func=np.mean):
    x = x - mean_func(x, axis=axis, keepdims=True)
    x = x / mean_func(x**2, axis=axis, keepdims=True) ** 0.5
    return x


def calc_tf_scores(x, y, spec_seg, spec_inc, gauss_kernel):
    x = normalize(x, axis=-1, mean_func=np.nanmean)
    y = normalize(y, axis=-1, mean_func=np.nanmean)

    x_seg = unfold(x, spec_seg, spec_inc, axis=1)
    y_seg = unfold(y, spec_seg, spec_inc, axis=1)

    with np.errstate(invalid="ignore"):
        x_seg = normalize(x_seg, axis=1)
        y_seg = normalize(y_seg, axis=1)
    U = np.mean(x_seg * y_seg, axis=1)

    # handle last segment
    n_last = spec_seg - spec_inc
    if n_last:
        with np.errstate(invalid="ignore"):
            x_seg = normalize(x[:, x.shape[1] - n_last :, None, :], axis=1)
            y_seg = normalize(y[:, y.shape[1] - n_last :, None, :], axis=1)
        U = np.concatenate([U, np.mean(x_seg * y_seg, axis=1)], axis=1)

    U = np.maximum(U, 0)
    return np.einsum("nij,ki->nkj", U, gauss_kernel)
