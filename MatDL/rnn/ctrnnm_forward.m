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
    % ACC1: 1 3 2 1 2 3 4 5 6 6
    % ACC2: 1 3 2 1 2 3 4 5 6 6
    % ACC3: 1 3 2 1 2 3 4 5 6 6
            %<------>
            %  <------>
            %     <------>
    
    [M, ~, T] = size(x); %input dimensions, ~ windows of M elements in each sliding window for T timesteps
    H = size(wh, 2); %size of hidden layer (number of neurons)
    out = zeros(M, H, T); % output of one forward pass
    
    %Voltage update equation
    %vprev=(repmat(vprev,1,M)*tanh(out(:, :, 1)-4.2000e-05)*wh)+repmat(b, H, 1); %V update equation on page 2

    C=(4.9311e-20./(4.2000e-05-hprev).^2); %a term of Equation (4) where hprev ~ z_i(t) 
    %vprev(Previous Voltage) is vector 5x1, Since we have 5 neurons, we will repeat it for 5 times to create a 5x5 matrix
    term1=(C.*(tanh(hprev-4.1800e-05)*wh*repmat(vprev,1,H))+repmat(b, M, 1)).^2; %a term of Equation (4) where vprev ~ V_j, wh ~ w_ij, b ~ theta_i
    term2=(4.1339e-08*(x(:, :, 1)*wx)); %a term of Equation (4) where x ~ accelerometer input, wx ~ w_in,k
    out(:, :, 1) = 0.9395*hprev+term1-term2; %equation 4 implemented
    v=(repmat(vprev,1,M)*tanh(out(:, :, 1)-4.2000e-05)*wh)+repmat(b, H, 1); %V update equation on page 2
    for t = 2:T
        C=(4.9311e-20./(4.2000e-05-out(:, :, t - 1)).^2); % an iterative version of C
         % (below) an iterative version of Equation 4
        out(:,:, t) = 0.9395*out(:, :, t - 1)+(C.*(tanh(out(:, :, t - 1)-4.1800e-05)*wh*v)+repmat(b, M, 1)).^2-(4.1339e-08*(x(:, :, t)*wx));
        %vprev=previous layer voltages, out(t) ~ z_(t)
        %dim(v)=49x5 or 176x5
        %v's should not go negative % v should be handled as z
        v=(repmat(vprev,1,M)*tanh(out(:, :, t)-4.2000e-05)*wh)+repmat(b, H, 1); %for every neuron V must be re-evaluated at every time step.
    end
    cache.vprev=v(1,:)'; %store v into cache so that it can be used during training passes
    cache.x = x; cache.hprev = hprev; cache.wx = wx; cache.wh = wh; cache.b = b; cache.out = out;
end