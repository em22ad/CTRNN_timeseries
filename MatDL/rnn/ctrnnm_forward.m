function [out, cache] = ctrnnm_forward(x, hprev, wx, wh, b, vprev)
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
    [M, ~, T] = size(x);
    H = size(wh, 2);
    out = zeros(M, H, T);
    C=(4.9311e-20./(4.2000e-05-hprev).^2);
    term1=(C.*(tanh(hprev-4.1800e-05)*wh*vprev)+repmat(b, M, 1)).^2;
    term2=(4.1339e-08*(x(:, :, 1)*wx));
    out(:, :, 1) = 0.9395*hprev+term1-term2;
    v=(tanh(hprev-4.2000e-05)*wh*vprev)+repmat(b, M, 1); %tanh(...)-> YYY x 5
    for t = 2:T
        C=(4.9311e-20./(4.2000e-05-out(:, :, t - 1)).^2);
        out(:, :, t) = 0.9395*out(:, :, t - 1)+(C.*(tanh(out(:, :, t - 1)-4.1800e-05)*wh*vprev)+repmat(b, M, 1)).^2-(4.1339e-08*(x(:, :, t)*wx));                
    end
    cache.vprev=v; 
    cache.x = x; cache.hprev = hprev; cache.wx = wx; cache.wh = wh; cache.b = b; cache.out = out;
end