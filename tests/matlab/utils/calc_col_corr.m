function col_corr = calc_col_corr(Xk, Yk)
%
% This function calculates correlation between columns of the 
% clean and degraded spectral segments Xk and Yk, respectively.
%
% INPUTS: 
%   Xk         : clean spectral segment
%   Yk         : degraded spectral segment
% 
% OUTPUTS: 
%   col_corr   : row vector of correlations of columns of Xk and Yk
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


% number cochlear channles in the spectral segment
J = size(Xk, 1);

% normalize columns to zero mean
Yk = Yk - ones(J,1)*mean(Yk, 'omitnan');
Xk = Xk - ones(J,1)*mean(Xk, 'omitnan');

% normalize cols to unit std
Yk = Yk.*repmat(1./sqrt(mean((Yk.^2), 'omitnan')), J, 1);
Xk = Xk.*repmat(1./sqrt(mean((Xk.^2), 'omitnan')), J, 1);

% calculate correlation between columns of Xk and Y
col_corr = mean(Yk .* Xk, 'omitnan');

% set nan values to 0
col_corr(isnan(col_corr)) = 0;

