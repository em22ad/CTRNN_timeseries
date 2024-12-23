function [dx, dw, db] = backward(dout, cache)
%BACKWARD Compute the gradients of a fully-connected layer
%   Inputs:
%       - dout: upstream derivatives, of size: batch size (m) x layer size (l)
%       - cache: a structure of:
%           x: input to layer
%           w: weights
%           b: biases
%   Outputs:
%       - dx: gradients w.r.t. x, of size: size(x)
%       - dw: gradients w.r.t. w, of size: size(w)
%       - db: gradients w.r.t. b, of size: size(b)
%
% This file is part of the MaTDL toolbox and is made available under the 
% terms of the MIT license (see the LICENSE file) 
% from http://github.com/haythamfayek/MatDL
% Copyright (C) 2016-17 Haytham Fayek.

    x = cache.x; w = cache.w; % b = cache.b;
    dx = dout * w';
    dw = x' * dout;
    db = sum(dout, 1);
end