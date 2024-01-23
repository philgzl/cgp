function  H = fft_cochlear_fb(fs, N_fft, cfs)
%   
% This function returns the cochlear filterbank matrix 
%
% INPUTS: 
%   fs         : sampling rate in Hz
%   N_fft      : fft size
%   cfs        : center frequencies of the cochlear bands
% 
% OUTPUTS: 
%   H          : cochlear fiterbank matrix
%
% 
% This file is part of cgp.
%
%
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

% number of channels in the cochlear filterbank
num_bands = length(cfs);

% frequency axis
f = linspace(0, fs, N_fft+1);
f = f(1:(N_fft/2+1));

% ERB parameters (Glasberg and Moore Parameters)
beta = 1;
ear_Q = 9.26449; 
min_BW = 24.7;   
ERB = beta*(cfs/ear_Q + min_BW);

%% fft-based cochlear filterbank

% lower boundaries of cochlear filters (assuming symmetry of cochlear filters)
fl = cfs - ERB/2;
fr = cfs + ERB/2;
H = zeros(num_bands, length(f));

for i = 1:num_bands
    [a b]       = min((f-fl(i)).^2);
    fl(i)       = f(b);
    il          = b;

	[a b]       = min((f-fr(i)).^2);
    fr(i)       = f(b);
    ir          = b-1;
    
    H(i,il:ir)  = 1;
end
