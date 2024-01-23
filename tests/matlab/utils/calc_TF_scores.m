function S = calc_TF_scores(X, Y, spc_seg, spc_inc, W)
%   
% This function calculate correlations between spectral segments and 
% performs Gaussian kernel interpolation to calculate TF scores. 
%
% INPUTS: 
%   x          : clean auditory representation 
%   y          : degraded auditory representation 
%   spc_seg    : spectral segment length 
%   spc_inc    : spectral segment increment
%   W          : Gaussian kernel matrix 
% 
% OUTPUTS: 
%   S          : TF scores 
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


[J,N_sig] = size(X);

spc_idx = 1:spc_inc:(J-spc_seg+1);
num_rows = length(spc_idx);

% normalize rows of Xc and Xd
% first normalize rows to zero mean
X = X - mean(X.', 'omitnan').'*ones(1, N_sig);
Y = Y - mean(Y.', 'omitnan').'*ones(1, N_sig);

% normalize rows to unit std
X = repmat(1./std(X, 1, 2, 'omitnan'), 1, N_sig).*X;
Y = repmat(1./std(Y, 1, 2, 'omitnan'), 1, N_sig).*Y;

% initialize matrix of spectral correlations
U = zeros(num_rows, N_sig);

% compute correlation over subbands
for j=1:num_rows
    % spectral windowing
    spc_st = spc_idx(j);
    spc_en = spc_st + spc_seg - 1;

    Xc_spc_seg = X(spc_st:spc_en, :);
    Xd_spc_seg = Y(spc_st:spc_en, :);

    U(j, :) = calc_col_corr(Xc_spc_seg, Xd_spc_seg);
end

% add scores of last spectral segment
N_last = spc_seg - spc_inc;
U(num_rows+1,:) = calc_col_corr(X(J-N_last+1:J, :), Y(J-N_last+1:J, :));

% set negative correlations to zero
U = max(0, U);

% calculate TF scores using Gaussian kernel interpolation 
S = W * U;


