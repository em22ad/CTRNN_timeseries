% A complete example of a continuous time recurrent neural network
%% Init
clear all
addpath(genpath('../MatDL'));
%% Generate NARMA Series
N=300;
Xin = gen_narma_30(N);
Xin=Xin(100:end);
%X = round(rand(1,1000));
T = [(Xin(2:end)-Xin(1:end-1) > 0)];

pattern=[1 0 1];
idxs=strfind(T,pattern);
Tp=zeros(1,size(Xin,2));
Tp(idxs)=1;

plot(1:size(Xin,2),Xin(1:end))
hold on;
keep_on=0;

for i=2:size(Xin,2)
    if (Tp(i) > 0)
        keep_on=size(pattern,2);
    end
    
    if (keep_on >= 0) 
        plot([i-1 i],[Xin(i-1) Xin(i)],'m-')
        keep_on = keep_on-1;
    end
end

X_f=[];
Y=[];
XVal_f=[];
YVal=[];
for i=1:size(Xin,2)-(size(pattern,2))
    if mod(i,3) ~= 0
        if (sum(pattern-T(i:i+size(pattern,2)-1)) == 0)
            X_f=[X_f;Xin(i:i+size(pattern,2))];
            Y=[Y;[1 0]];
        else
            X_f=[X_f;Xin(i:i+size(pattern,2))];
            Y=[Y;[0 1]];
        end
    else
        if (sum(pattern-T(i:i+size(pattern,2)-1)) == 0)
            XVal_f=[XVal_f;Xin(i:i+size(pattern,2))];
            YVal=[YVal;[1 0]];
        else
            XVal_f=[XVal_f;Xin(i:i+size(pattern,2))];
            YVal=[YVal;[0 1]];
        end
    end  
end

for idx=1:size(X_f,1)
    X(idx,:,1)=X_f(idx,:);
end

for idx=1:size(XVal_f,1)
    XVal(idx,:,1)=XVal_f(idx,:);
end

%% Load data
% load('../Data/mnist_uint8.mat');
% %X = double(reshape(train_x',28,28,60000))/255;
% X = double(reshape(train_x',8,98,60000))/255;
% X = permute(X, [3 2 1]);
% %XVal = double(reshape(test_x',28,28,10000))/255;
% XVal = double(reshape(test_x',8,98,10000))/255;
% XVal = permute(XVal, [3 2 1]);
% Y = double(train_y);
% YVal = double(test_y);
%% Initialize model
opt = struct;
TS=0.01;
layers_size=[5 5];
[model, opt] = init_two_ctrnn(size(X,2), size(Y,2), layers_size, opt);

%% Hyper-parameters
opt.batchSize = 1;

opt.optim = @rmsprop;
% opt.beta1 = 0.9; opt.beta2 = 0.999; opt.t = 0; opt.mgrads = opt.vgrads;
opt.rmspropDecay = 0.75;
% opt.initialMomentum = 0.5; opt.switchEpochMomentum = 1; opt.finalMomentum = 0.9;
opt.learningRate = 0.1;
opt.learningDecaySchedule = 'exp'; %'no_decay';%'t/T';%'step';
opt.learningDecayRate = 1;
%opt.learningDecayRateStep = 5;

opt.dropout = 1;
opt.weightDecay = false;
opt.maxNorm = false;

opt.maxEpochs = 50;
opt.earlyStoppingPatience = 5;
opt.valFreq = 100;

opt.plotProgress = false;
opt.extractFeature = false;
opt.computeDX = false;
opt.t= randn(1, layers_size(1));
opt.useGPU = false;
if (opt.useGPU) % Copy data, dropout, model, vgrads, BNParams
    X = gpuArray(X); Y = gpuArray(Y); XVal = gpuArray(XVal); YVal = gpuArray(YVal); 
    opt.dropout = gpuArray(opt.dropout);
    p = fieldnames(model);
    for i = 1:numel(p), model.(p{i}) = gpuArray(model.(p{i})); opt.vgrads.(p{i}) = gpuArray(opt.vgrads.(p{i})); end
    p = fieldnames(opt);
    for i = 1:numel(p), if (strfind(p{i},'bnParam')), opt.(p{i}).runningMean = gpuArray(opt.(p{i}).runningMean); opt.(p{i}).runningVar = gpuArray(opt.(p{i}).runningVar); end; end
end

%% Gradient check
x = X(1:size(X,1)/2, :, :);
y = Y(1:size(Y,1)/2, :, :);
maxRelError = gradcheck(@two_ctrnn, x, model, y, opt, size(Y,2)); %last argument should be less or equal to the number of classes

%% Train
[model, trainLoss, trainAccuracy, valLoss, valAccuracy, opt] = train( X, Y, XVal, YVal, model, @two_ctrnn, opt );

%% Predict
[yplabel, confidence, classes, classConfidences, yp] = predict(XVal, @two_ctrnn, model, opt);
%
