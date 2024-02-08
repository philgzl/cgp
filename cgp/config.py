import numpy as np

fs = 10000
vad_win_len = round(25.6e-3 * fs)
vad_hop_len = round(12.8e-3 * fs)
vad_dyn_range = 40
stft_win_len = round(24e-3 * fs)
stft_hop_len = round(18e-3 * fs)
stft_n_fft = 1024
n_octaves = 5
channels_per_octave = 12
n_channels = n_octaves * channels_per_octave
min_fc = 150
fcs = min_fc * 2 ** (np.arange(n_channels) / channels_per_octave)
compression_factor = 0.1414
lpf_fc = 5.0526
spec_seg_len = 12
spec_seg_hop = 4
gauss_kernel_width = 4
env_ratio_thr = -1.5005
glimpse_thrs = [0.36461, 0.58204, 0.00709, 0.00867, 0.59439, 0.33654]
thr_channels = [0, 11, 23, 35, 47, 59]
