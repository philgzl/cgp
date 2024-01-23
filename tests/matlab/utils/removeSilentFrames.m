function [x_sil, y_sil] = removeSilentFrames(x, y, range, N, K)
% [X_SIL Y_SIL] = REMOVESILENTFRAMES(X, Y, RANGE, N, K) X and Y
% are segmented with frame-length N and overlap K, where the maximum energy
% of all frames of X is determined, say X_MAX. X_SIL and Y_SIL are the
% reconstructed signals, excluding the frames, where the energy of a frame
% of X is smaller than X_MAX-RANGE
%
%
% This file is part of cgp.
%
% Copyright 2016: Aalborg University, Section for Signal and Information Processing. 
% The software is free for non-commercial use. 
% The software comes WITHOUT ANY WARRANTY.
%


x       = x(:);
y       = y(:);

frames  = 1:K:(length(x)-N);
w       = hanning(N);
msk     = zeros(size(frames));

for j = 1:length(frames)
    jj      = frames(j):(frames(j)+N-1);
    msk(j) 	= 20*log10(norm(x(jj).*w)./sqrt(N));
end

msk     = (msk-max(msk)+range)>0;
count   = 1;

x_sil   = zeros(size(x));
y_sil   = zeros(size(y));

for j = 1:length(frames)
    if msk(j)
        jj_i            = frames(j):(frames(j)+N-1);
        jj_o            = frames(count):(frames(count)+N-1);
        x_sil(jj_o)     = x_sil(jj_o) + x(jj_i).*w;
        y_sil(jj_o)  	= y_sil(jj_o) + y(jj_i).*w;
        count           = count+1;
    end
end

x_sil = x_sil(1:jj_o(end));
y_sil = y_sil(1:jj_o(end));
