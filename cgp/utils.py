import numpy as np

EPS = np.finfo("float").eps


def remove_silent_frames(x, y, dyn_range, win_len, hop_len):
    # Taken from https://github.com/mpariente/pystoi
    # Copyright (c) 2018 Pariente Manuel
    # MIT License
    w = np.hanning(win_len + 2)[1:-1]

    x_frames = np.array(
        [w * x[i:i + win_len] for i in range(0, len(x) - win_len, hop_len)]
    )
    y_frames = np.array(
        [w * y[i:i + win_len] for i in range(0, len(x) - win_len, hop_len)]
    )

    x_energies = 20 * np.log10(np.linalg.norm(x_frames, axis=1) + EPS)

    mask = (np.max(x_energies) - dyn_range - x_energies) < 0

    x_frames = x_frames[mask]
    y_frames = y_frames[mask]

    x_sil = _overlap_and_add(x_frames, hop_len)
    y_sil = _overlap_and_add(y_frames, hop_len)

    return x_sil, y_sil


def _overlap_and_add(x_frames, hop_len):
    # Taken from https://github.com/mpariente/pystoi
    # Copyright (c) 2018 Pariente Manuel
    # MIT License
    # Original PR by Gianluca Micchi
    # https://github.com/mpariente/pystoi/pull/26
    n_frames, win_len = x_frames.shape

    # Compute the number of segments per frame
    n_seg = -(-win_len // hop_len)  # Divide and round up

    # Pad the win_len dimension to n_seg * hop_len and add n_seg frames
    signal = np.pad(x_frames, ((0, n_seg), (0, n_seg * hop_len - win_len)))

    # Reshape to a 3D tensor, splitting the win_len dimension in two
    signal = signal.reshape((n_frames + n_seg, n_seg, hop_len))

    # Transpose so that signal.shape = (n_seg, n_frames + n_seg, hop_len)
    signal = np.transpose(signal, [1, 0, 2])

    # Reshape so that signal.shape = (n_seg * (n_frames + n_seg), hop_len)
    signal = signal.reshape((-1, hop_len))

    # Now behold the magic! Remove the last n_seg elements from the first axis
    signal = signal[:-n_seg]

    # Reshape to (n_seg, n_frames + n_seg - 1, hop_len)
    signal = signal.reshape((n_seg, n_frames + n_seg - 1, hop_len))
    # This has introduced a shift by one in all rows

    # Now reduce over the columns and flatten the array to achieve the result
    signal = np.sum(signal, axis=0)
    end = (len(x_frames) - 1) * hop_len + win_len
    signal = signal.reshape(-1)[:end]

    return signal


def gen_cochlear_fb(fs, n_fft, cfs):
    """Generate a cochlear filterbank.

    The output is a matrix `fb` of size (len(cfs), n_fft//2 + 1) such that
    `fb[i, j] = 1` if the j-th frequency bin is in the i-th cochlear filter,
    and `fb[i, j] = 0` otherwise.

    In other words, if `f_j` is the j-th frequency bin, `f_l_i` is the lower
    bound of the i-th cochlear filter and `f_r_i` is the upper boundary of the
    i-th cochlear filter, then `fb[i, j] = 1` if `f_l_i <= f_j <= f_r_i` and
    `fb[i, j] = 0` otherwise.

    """
    # Frequency axis
    f = np.arange(n_fft//2 + 1) * fs / n_fft

    # ERB parameters (Glasberg and Moore)
    beta = 1
    ear_q = 9.26449
    min_bw = 24.7
    erb = beta*(cfs/ear_q + min_bw)

    # Lower boundaries of cochlear filters
    fl = cfs - erb/2
    fr = cfs + erb/2

    fb = np.zeros((len(cfs), len(f)))
    for i in range(len(cfs)):
        # In the original code the filterbank is implemented by finding the
        # closest frequency bin to the lower and upper boundaries and then
        # placing ones between the boundaries. This is:
        # 1. Not accurate as the closest frequency bin might be left of the
        #    lower boundary or right of the upper boundary which means that
        #    the filters might be wider than intended
        # 2. Not efficient as it does not take advantage of the fact that
        #    the frequency bins and boundaries are sorted

        # Find frequency bin closest to lower boundary
        il = np.argmin(np.abs(f - fl[i]))

        # Find frequency bin closest to upper boundary
        ir = np.argmin(np.abs(f - fr[i]))

        # Place ones between lower and upper boundary
        fb[i, il:ir + 1] = 1


def gen_tmp_mod_lpf(tmp_cf, tmp_fs, tmp_mod_nfft):
    pass


def calc_tf_scores(x, y, spec_seg, spec_inc, gauss_kernel):
    pass
