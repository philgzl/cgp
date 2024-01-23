function d = cgp(x, y, fs_sig)
%
% This function returns the correlation based glimpse proportion (CGP) 
% index described in [1]. 
%
% INPUTS: 
%   x          : clean speech
%   y          : degraded speech
%   fs_sig     : sampling rate of x and y in Hz
% 
% OUTPUTS: 
%   d          : overall CGP score
%
%
% References: 
%  [1] A. Alghamdi, L. Moen, W. Y. Chan, D. Fogerty and J. Jensen, "Correlation based 
%      glimpse proportion index," WASPAA 2023.
%
% Copyright 2023, Ahmed Alghamdi, a.alghamdi@queensu.ca, Queen's University
% at Kingston, Multimedia Coding and Communications Laboratory. 
% 
%
% cgp is free software: you can redistribute it and/or modify it under the 
% terms of the GNU General Public License as published by the Free Software 
% Foundation; either version 3 of the License, or (at your option) any 
% later version.
% 
% cgp is distributed in the hope that it will be useful, but WITHOUT ANY 
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
% FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
% details.
% 
% You should have received a copy of the GNU General Public License along 
% with this program. If not, see https://www.gnu.org/licenses/. 

%% Initialization 

fs = 10e3;                                              % sampling of the objective measure
win_lenS = 24e-3;                                       % window length in seconds 
win_incS = 18e-3;                                       % window increment in seconds 
ac_fft_len = 1024;                                      % fft size used in STFT decomposition
min_ac_cf = 150;                                        % minimum cochlear center frequency
spc_res = 12;                                           % spectral resolution in cycle/octave
num_ac_octaves = 5;                                     % number of octaves covered by cochlear filterbank
J = num_ac_octaves*spc_res;                             % number of channles in the cochlear filterbank
ac_cfs = min_ac_cf*2.^((0:J-1)/spc_res);                % center frequencies of cochlear filterbnak
alpha = 0.1414;                                         % fractional exponent
tmp_cf = 5.0526;                                        % cutoff frequency of temporal envelope LPF filter  
spc_seg = 12;                                           % spectral segment length (full octave segment)
spc_inc = 4;                                            % spectral segment increment (1/3 octave increments) 
GBW = 4;                                                % BW of Gaussian kernel used in spectral scores interpolation
delta = -1.5005;                                        % threshold on envelope ratio to identify energetic regions of clean envelopes
z = [0.36461 0.58204 0.00709 0.00867 0.59439 0.33654];  % ocatve-spaced glimpsing thresholds

% parameters of vad dectector to remove inactive regions from the clean and degraded signals
vad_win_lenS = 25.6e-3;                           % window length in seconds
vad_win_incS = 12.8e-3;                           % window increment in seconds
vad_dyn_range = 40;                               % dynamic range in dB

%% pre-processing

% check if the input signals have the same length
if length(x) ~= length(y)
    error('Clean and degraded signals must have the same length.');
end

% resample signals if necessary
if fs_sig ~= fs
    x = resample(x, fs, fs_sig);
    y = resample(y, fs, fs_sig);
end

% remove inactive regions from the clean and degraded signals
vad_win_len = round(vad_win_lenS * fs);        
vad_win_inc = round(vad_win_incS * fs);   
[x, y] = removeSilentFrames(x, y, vad_dyn_range, vad_win_len, vad_win_inc); 


%% cochlear filterbank and envelope extraction

% window length in samples
win_len = round(win_lenS * fs);

% window increment length in samples
win_inc = round(win_incS * fs);

% calculate single-sided short-time DFT of clean signal 
X_stft = stdft(x, win_len, win_inc, ac_fft_len);

% calculate single-sided short-time DFT of degraded siganl 
Y_stft = stdft(y, win_len, win_inc, ac_fft_len);

% construct fft-based cochlear filterbank
H_cfb = gen_cochlear_fb(fs, ac_fft_len, ac_cfs);  

% apply the FFT cochlear filterbank
X = sqrt((H_cfb.^2) * (abs(X_stft).^2));
Y = sqrt((H_cfb.^2) * (abs(Y_stft).^2));

% amplitude compression
X = X.^alpha;
Y = Y.^alpha;

% calculate fft size used in temporal modulation filtering
[J, N_sig] = size(X);
tmp_mod_nfft = 2^nextpow2(N_sig);

% construct LPF temporal modulation filter
tmp_fs = 1/win_incS;
Ht = gen_tmp_mod_LPF(tmp_cf, tmp_fs, tmp_mod_nfft);
Ht = Ht(:)';

% fft of rows of X and Y
fft_c = fft(X, tmp_mod_nfft, 2);
fft_d = fft(Y, tmp_mod_nfft, 2);

% apply temporal modulation filter
X = real(ifft(Ht .* fft_c, tmp_mod_nfft, 2));
X = X(:,1:N_sig);

Y = real(ifft(Ht .* fft_d, tmp_mod_nfft, 2));
Y = Y(:,1:N_sig);

%% similarity computations 

% starting positions of spectral segments
spc_idx = 1:spc_inc:(J-(spc_seg-spc_inc)+1);

% generate Gaussian weighting matrix
sigma = GBW*ones(1, length(spc_idx));
beta = 1 ./ (2 * sigma.^2);
x = 1:J;
W = zeros(length(spc_idx), J);
for j=1:length(spc_idx)
    W(j,:) = exp(-beta(j) .* (x - spc_idx(j)).^2);
end

W = W';
W = W ./ sum(W, 2);

% calculate TF scores 
S = calc_TF_scores(X, Y, spc_seg, spc_inc, W);

% calculate rms value of temporal envelopes across spectral channels
env_rms = sqrt(mean(X.^2, 2));

% calculate envelope ratio in dB
ratio_db = 20*log10(X ./ env_rms);

% 
ER_mask = ratio_db >= delta;

% positions of optimized glimpsing thresholds
spc_pts = [1 12 24 36 48 60];                     

% calculate frquency dependent glimpsing thresholds
gamma = interp1(spc_pts, z, 1:J, "linear");       

% glimpsing of TF scores
G = S - gamma(:);
G(G<=0) = 0;
G(G>0) = 1;

% calculate final score as GP over energetic regions of clean envelopes
d = mean(G(ER_mask)); 
