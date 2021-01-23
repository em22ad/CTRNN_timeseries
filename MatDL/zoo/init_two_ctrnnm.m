function [model, opt] = init_two_ctrnnm(M, N, K, layers_size, opt)
    
    %M=round(M/2.0);
    weightScale = 0.01;
    biasScale = 100;

    NL1 = layers_size(1);

    model.wx1 = randn(N, NL1) * weightScale;
    model.wh1 = randn(NL1, NL1) * weightScale;
    model.b1 = randn(1, NL1) * biasScale;
    model.wy = randn(NL1, K) * weightScale;
    model.by = rand(1, K) * biasScale; %bias cannot be negative
    %model.v = zeros(NL1,M); %each element of v corresponds to voltage associated to each neuron
    
    p = fieldnames(model);
    for i = 1:numel(p)
        opt.vgrads.(p{i}) = zeros(size(model.(p{i})));
    end

end