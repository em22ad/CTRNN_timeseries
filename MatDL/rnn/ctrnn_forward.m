function [out, cache] = ctrnn_forward(x, hprev, wx, wh, b)
%RNN_FORWARD Compute the forward pass for a recurrent layer
%   Inputs:
%       - x: input to layer, of size: batch size (m) x previous layer size x time steps (t) 
%       - hprev: hidden state from previous (initial) time step, of size: batch size (m) x layer size (l)
%       - wx: input-to-hidden weights, of size: previous layer size x layer size (l)
%       - wh: hidden-to-hidden weights, of size: layer size (l) x layer size (l)
%       - b: biases, of size: 1 x layer size (l)
%   Outputs:
%       - out: output, of size: batch size (m) x layer size (l) x time steps (t)
%       - cache: a structure of:
%           x: input to layer
%           hprev: initial hidden state
%           wx: input-to-hidden weights
%           wh: hidden-to-hidden weights
%           b: biases
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.
    ts=0.3;%0.035;
    [M, ~, T] = size(x);
    H = size(wh, 2);
    out = zeros(M, H, T);
    %size(hprev)
    %out(:, :, 1) = tanh( x(:, :, 1) * wx + hprev * wh + repmat(b, M, 1) );
    out(:, :, 1) = (1-ts)*hprev+ts*(tanh( x(:, :, 1) * wx + hprev * wh) + repmat(b, M, 1));
    for t = 2:T
        %out(:, :, t) = tanh( x(:, :, t) * wx + out(:, :, t - 1) * wh + repmat(b, M, 1) );
        out(:, :, t) = (1-ts)*out(:, :, t - 1)+ts*(tanh( x(:, :, t) * wx + out(:, :, t - 1) * wh) + repmat(b, M, 1) );
        %out(:, :, t) = 0.9395*out(:, :, t - 1)+((C(out(:, :, t - 1)) * wx * V * sigma(hprev)) + repmat(b, M, 1))^2 
    end
    cache.x = x; cache.hprev = hprev; cache.wx = wx; cache.wh = wh; cache.b = b; cache.out = out;
end