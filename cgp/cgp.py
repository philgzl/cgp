import numpy as np
import scipy.signal

from . import utils

FS = 10000
WIN_LEN_S = 24e-3
WIN_INC_S = 18e-3
AC_FFT_LEN = 1024
MIN_AC_CF = 150
SPEC_RES = 12
NUM_AC_OCT = 5
N_CHANNELS = NUM_AC_OCT * SPEC_RES
AC_CFS = MIN_AC_CF * 2 ** (np.arange(N_CHANNELS) / SPEC_RES)
ALPHA = 0.1414
TMP_CF = 5.0526
SPEC_SEG = 12
SPEC_INC = 4
GBW = 4
DELTA = -1.5005
GLIMPSE_THR = [0.36461, 0.58204, 0.00709, 0.00867, 0.59439, 0.33654]
VAD_WIN_LEN_S = 25.6e-3
VAD_WIN_INC_S = 12.8e-3
VAD_DYN_RANGE = 40


def cgp(x, y, fs):
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError('x and y must be 1D arrays.')
    if x.shape != y.shape:
        raise ValueError('x and y must have the same shape.')

    # resample
    if fs != FS:
        x = scipy.signal.resample_poly(x, FS, fs)
        y = scipy.signal.resample_poly(y, FS, fs)

    # VAD
    vad_win_len = round(VAD_WIN_LEN_S * FS)
    vad_win_inc = round(VAD_WIN_INC_S * FS)
    x, y = utils.remove_silent_frames(x, y, VAD_DYN_RANGE, vad_win_len,
                                      vad_win_inc)

    # STFT
    win_len = round(WIN_LEN_S * FS)
    win_inc = round(WIN_INC_S * FS)
    _, _, x_stft = scipy.signal.stft(x, nperseg=win_len,
                                     noverlap=win_len - win_inc,
                                     nfft=AC_FFT_LEN)
    _, _, y_stft = scipy.signal.stft(y, nperseg=win_len,
                                     noverlap=win_len - win_inc,
                                     nfft=AC_FFT_LEN)

    # FFT-based cochlear filterbank
    h_cfb = utils.gen_cochlear_fb(FS, AC_FFT_LEN, AC_CFS)
    x = h_cfb.dot(x_stft.abs())
    y = h_cfb.dot(y_stft.abs())

    # amplitude compression
    x **= ALPHA
    y **= ALPHA

    # temporal modulation filtering
    tmp_mod_nfft = 2 ** np.ceil(np.log2(x.shape[-1]))
    tmp_fs = 1 / WIN_INC_S
    h_t = utils.gen_tmp_mod_lpf(TMP_CF, tmp_fs, tmp_mod_nfft)
    fft_c = np.fft.fft(x, tmp_mod_nfft)
    fft_d = np.fft.fft(y, tmp_mod_nfft)

    x = np.fft.ifft(h_t * fft_c, tmp_mod_nfft).real[:, :x.shape[-1]]
    y = np.fft.ifft(h_t * fft_d, tmp_mod_nfft).real[:, :y.shape[-1]]

    # calculate TF scores
    spc_idx = np.arange(0, x.shape[0] - SPEC_SEG + 1, SPEC_INC)
    jj = np.arange(N_CHANNELS)
    gauss_kernel = (-(np.subtract.outer(jj, spc_idx) / (2 * GBW)) ** 2).exp()
    gauss_kernel /= gauss_kernel.sum(axis=1, keepdims=True)
    scores = utils.calc_tf_scores(x, y, SPEC_SEG, SPEC_INC, gauss_kernel)

    # glimpsing of TF scores
    env_rms = (x ** 2).mean(axis=1, keepdims=True) ** 0.5
    ratio_db = 20 * np.log10(x / env_rms)
    er_mask = ratio_db >= DELTA
    spc_pts = [1, 12, 24, 36, 48, 60]
    gamma = np.interp(jj, spc_pts, GLIMPSE_THR)
    g = np.clip(scores - gamma, 0, 1)

    # final scores as GP over energetic regions of clean envelopes
    return (g * er_mask).mean()
