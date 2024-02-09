import numpy as np
import scipy.signal


def unfold(x, win_len, hop_len):
    n_frames = (len(x) - win_len) // hop_len + 1
    idx = np.add.outer(np.arange(win_len), np.arange(n_frames) * hop_len)
    return x[idx]


def hanning(n):
    win = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, n + 1) / (n + 1)))
    return win[:, None]


def stft(x, win_len, hop_len, n_fft):
    x = unfold(x, win_len, hop_len) * hanning(win_len)
    return np.fft.rfft(x, n=n_fft, axis=0)


def resample(x, fs_in, fs_out):
    return scipy.signal.resample_poly(x, fs_out, fs_in)


def remove_silent_frames(x, y, win_len, hop_len, dyn_range, _discard_last_frame=False):
    # Inspired from https://github.com/mpariente/pystoi
    # Copyright (c) 2018 Pariente Manuel
    # MIT License
    x_frames = unfold(x, win_len, hop_len) * hanning(win_len)
    y_frames = unfold(y, win_len, hop_len) * hanning(win_len)

    # The original MATLAB implementation discards the last frame if the signal
    # fits an integer number of frames
    if _discard_last_frame and (x.shape[-1] - win_len) % hop_len == 0:
        x_frames = x_frames[..., :-1]
        y_frames = y_frames[..., :-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        x_dB = 20 * np.log10(np.linalg.norm(x_frames, axis=0))
        mask = (np.max(x_dB) - dyn_range - x_dB) < 0

    if not any(mask):
        raise RuntimeError("No speech frames detected")

    x = overlap_and_add(x_frames[:, mask], hop_len)
    y = overlap_and_add(y_frames[:, mask], hop_len)
    return x, y


def overlap_and_add(x, hop_len):
    # Inspired from https://github.com/mpariente/pystoi
    # Copyright (c) 2018 Pariente Manuel
    # MIT License
    # Original PR by Gianluca Micchi
    # https://github.com/mpariente/pystoi/pull/26
    win_len, n_frames = x.shape

    # Compute the number of segments per frame
    n_seg = -(-win_len // hop_len)  # Divide and round up

    # Pad the win_len dimension to n_seg * hop_len and add n_seg frames
    x = np.pad(x, ((0, n_seg * hop_len - win_len), (0, n_seg)))

    # Reshape to a 3D tensor, splitting the win_len dimension in two
    x = x.reshape(n_seg, hop_len, n_frames + n_seg)

    # Transpose so that x.shape = (n_seg, n_frames + n_seg, hop_len)
    x = np.transpose(x, [0, 2, 1])

    # Reshape so that x.shape = (n_seg * (n_frames + n_seg), hop_len)
    x = x.reshape(-1, hop_len)

    # Now behold the magic! Remove the last n_seg elements from the first axis
    x = x[:-n_seg]

    # Reshape to (n_seg, n_frames + n_seg - 1, hop_len)
    x = x.reshape((n_seg, n_frames + n_seg - 1, hop_len))
    # This has introduced a shift by one in all rows

    # Now reduce over the columns and flatten the array to achieve the result
    x = np.sum(x, axis=0)
    end = (n_frames - 1) * hop_len + win_len
    return x.reshape(-1)[:end]


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


def apply_lpf(x, lpf):
    x_fft = np.fft.fft(x, len(lpf), axis=-1)
    return np.fft.ifft(lpf * x_fft, axis=-1).real[:, : x.shape[-1]]


def n_spec_seg(n_channels, spec_seg_len, spec_seg_hop):
    # The paper states "All segments have a length of full octave band or 12
    # channels except the last segment which has 8 channels". This means an
    # additional segment consisting of the last 8 channels is added even though
    # the last 8 channels fit in the last 12-channel segment. The start index
    # of each segment is calculated as follows in the MATLAB implementation:
    #
    #    spc_idx = 1:spc_inc:(J-(spc_seg-spc_inc)+1);
    #
    # which results in the following number of segments.
    return np.ceil(
        (n_channels - spec_seg_len + spec_seg_hop + 1) / spec_seg_hop
    ).astype(int)


def gauss_kernel(n_channels, spec_seg_len, spec_seg_hop, width):
    n_seg = n_spec_seg(n_channels, spec_seg_len, spec_seg_hop)
    seg_idx = np.arange(n_seg) * spec_seg_hop
    j = np.arange(n_channels)
    gauss_kernel = np.exp(-0.5 * (np.subtract.outer(j, seg_idx) / width) ** 2)
    return gauss_kernel / gauss_kernel.sum(axis=1, keepdims=True)


def normalize(x, axis):
    x = x - np.mean(x, axis=axis, keepdims=True)
    x = x / np.mean(x**2, axis=axis, keepdims=True) ** 0.5
    return x


def calc_tf_scores(x, y, spec_seg, spec_inc, gauss_kernel):
    x = normalize(x, axis=1)
    y = normalize(y, axis=1)

    n_seg = n_spec_seg(x.shape[0], spec_seg, spec_inc)
    idx = np.add.outer(np.arange(spec_seg), np.arange(n_seg - 1) * spec_inc)

    x_seg = normalize(x[idx, :], axis=0)
    y_seg = normalize(y[idx, :], axis=0)
    U = np.mean(x_seg * y_seg, axis=0)

    # handle last segment
    n_last = spec_seg - spec_inc
    if n_last:
        x_seg = normalize(x[x.shape[0] - n_last :, None, :], axis=0)
        y_seg = normalize(y[y.shape[0] - n_last :, None, :], axis=0)
        U = np.concatenate([U, np.mean(x_seg * y_seg, axis=0)], axis=0)

    U = np.maximum(U, 0)
    return np.dot(gauss_kernel, U)
