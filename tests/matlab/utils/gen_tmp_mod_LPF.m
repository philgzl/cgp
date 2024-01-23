function H = gen_tmp_mod_LPF(cf, tmp_fs, nfft)
% 
% This function constructs a lowpass temporal modulation filter operting
% in the DFT domain. The shape of the constructed filter was proposed by 
% Nemala et al. in [1]. 
%
% [1] Nemala SK, Patil K, Elhilali M., "A Multistream Feature Framework Based 
%     on Bandpass Modulation Filtering for Robust Speech Recognition", IEEE 
%     Trans Audio Speech Lang Process. 2013 February.
%
% INPUTS: 
%   cf         : cutoff frequency of LPF (must be less than f_max)
%   tmp_fs     : temporal modulation sampling rate in Hz 
%   nfft       : fft size used to implement the temporal modulation filter 
% 
% OUTPUTS: 
%   H          : output vector represents the magnitude response of the LPF filter 
%                with a length of nfft
%
% This file is part of cgp.
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


% define filter magnitude response 
mag_response = @(a, w) (a.*w).^2 .* exp(1 - (a.*w).^2);

% calculate frequency spacing
ds = tmp_fs / nfft;

% construct frequency axis
freq_axis = (0:(nfft-1))*ds;

% move all frequencies greater than tmp_fs/2 to the negative side of the axis
freq_axis(freq_axis>tmp_fs/2) = freq_axis(freq_axis>tmp_fs/2)-tmp_fs;

% set values of alpha 
alpha = zeros(size(freq_axis));
alpha(abs(freq_axis)<=cf) = 1./freq_axis(abs(freq_axis)<=cf);
alpha(abs(freq_axis)>cf) = 1./cf;

% construct filter response (zero phase) 
H = mag_response(alpha, freq_axis);

% set response value at 0Hz to unity
H(1) = 1;

% normalize to have a gain of 1 in the passband
H = H / max(H);



