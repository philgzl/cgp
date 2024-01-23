function x_stdft = stdft(x, N, K, N_fft)
% X_STDFT = X_STDFT(X, N, K, N_FFT) returns the short-time
% hanning-windowed dft of X with frame size N, hop size K and 
% DFT size N_FFT. The columns and rows of x_stdft denote the 
% dft bin index and frame index, respectively.
%
% This function was adapted from the ESTOI code implementation 
% 
%
% This file is part of cgp.
%
% 
%
% Copyright 2016: Aalborg University, Section for Signal and Information Processing. 
% The software is free for non-commercial use. The software comes WITHOUT ANY WARRANTY.

% Copyright 2023: Ahmed Alghamdi, a.alghamdi@queensu.ca, Queen's University
% at Kingston, Multimedia Coding and Communications Laboratory. 



% convert input x to a column vector
x = x(:);
frames = 1:K:(length(x)-N+1);
w = hanning(N);
sig_frames = zeros(N, length(frames));

for j=1:length(frames)
    sig_frames(:,j) = x(frames(j):frames(j)+N-1).*w;
end

% calculate fft of frames
x_stdft = fft(sig_frames, N_fft);

% single-sided spectrum 
x_stdft = x_stdft(1:(N_fft/2+1), :);