function [out, cache] = ctrnnm_forward(x, hprev, wx, wh, b)
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
    omg=1209.5*2.0;
    zeta=1.028075;
    tau=(2*zeta)/omg;%0.0017;
    dt=0.0001;
    %M is the number of windows
    [M, ~, T] = size(x); %~M is the batch size for sliding window for T timesteps
    H = size(wh, 2); %size of hidden layer (number of neurons)
    out = zeros(M, H, T); % output of one forward pass
    
    %We always initialize the Voltage to be zeros.
    v=zeros(H,M);
    
    for wi=1:M
        %C=(4.9311e-20./(4.2000e-05-hprev(wi,:)).^2); %a term of Equation (4) where hprev ~ z_i(t)
        C=((8.1539e-19*(dt/tau))./(4.2000e-05-hprev(wi,:)).^2); %a term of Equation (4) where hprev ~ z_i(t)
        %vprev(Previous Voltage) is vector 5x1, Since we have 5 neurons, we will repeat it for 5 times to create a 5x5 matrix
        %(C(1x5).x tanh(z(1x5))w(5x1)*V(1*5))+ b(1x5)
        term1=C.*((tanh(hprev(wi,:)-4.1800e-05)*wh*v(:,wi))+b).^2; %a term of Equation (4) where vprev ~ V_j, wh ~ w_ij, b ~ theta_i
        %term2=(4.1339e-08*(x(wi, :, 1)*wx)); %a term of Equation (4) where x ~ accelerometer input, wx ~ w_in,k
        term2=(((dt/tau)*(1/omg^2))*(x(wi, :, 1)*wx)); %a term of Equation (4) where x ~ accelerometer input, wx ~ w_in,k
        %out(wi, :, 1) = 0.9395*hprev(wi,:)+term1-term2; %equation 4 implemented
        out(wi, :, 1) = (1-(dt/tau))*hprev(wi,:)+term1-term2; %equation 4 implemented
        out(out > 3.8000e-5)=3.8000e-5;%default=3.0000e-5
        v(:,wi)=(tanh(out(wi, :, 1)-4.000e-05)*wh*v(:,wi))+b; %V update equation on page 2
%         for j=1:M
%             for k=1:size(v,1)
%                 if (v(k,j) < 0)
%                     v(k,j)=abs(v(k,j));
%                 end
%                 %v(v < 0)=abs(v);%0;
%             end
%         end
        v(v < -140)=-140;
        v(v > 140)=140;
    end

    %176x49 %176 is the observation per window and 49 is the number of windows 
    %V(5x49)= V(5*49)*tanh(Z(49 x 5)*wh(5x5)+b(5x5)
    %v(5x1)=tanh(out(1x5))*wh(5x5)*v(5x1)'+b(1x5)
    %bias for V should all be positive
    %for wi=1:M
    %    v(:,wi)=(tanh(out(wi, :, 1)-4.2000e-05)*wh*vprev(:,wi))+b; %V update equation on page 2
    %end
    for wi=1:M
        for t = 2:T
            C=((8.1539e-19*(dt/tau))./(4.2000e-05-out(wi, :, t - 1)).^2); %a term of Equation (4) where hprev ~ z_i(t) 
            term1=C.*((tanh(out(wi, :, t - 1)-4.1800e-05)*wh*v(:,wi))+b).^2; %a term of Equation (4) where vprev ~ V_j, wh ~ w_ij, b ~ theta_i
            term2=(((dt/tau)*(1/omg^2))*(x(wi, :, t)*wx)); %a term of Equation (4) where x ~ accelerometer input, wx ~ w_in,k
            out(wi, :, t) = (1-(dt/tau))*out(wi, :, t - 1)+term1-term2; %equation 4 implemented
            out(out(wi, :, t) > 3.8000e-5)=3.8000e-5; %default: 3.0000e-5, upto 4.2/3(lower) 4.2000e-5(higher), smaller number means more history and vice versa
            v(:,wi)=(tanh(out(wi, :, t)-4.2000e-05)*wh*v(:,wi))+b; %V update equation on page 2
%             for j=1:M
%                 for k=1:size(v,1)
%                     if (v(k,j) < 0)
%                         v(k,j)=abs(v(k,j));
%                     end
%                     %v(v < 0)=abs(v);%0;
%                 end
%             end
            v(v < -140)=-140;
            v(v > 140)=140;
        end
        %for wi=1:M
        %    v(:,wi)=(tanh(out(wi, :, t)-4.2000e-05)*wh*vprev(:,wi))+b; %V update equation on page 2
        %end
        %vprev=v;
    end
    cache.x = x; 
    cache.hprev = hprev; 
    cache.wx = wx; 
    cache.wh = wh; 
    cache.b = b; 
    cache.out = out;
end