% A complete example of a continuous time recurrent neural network
%% Init
clear all;
addpath(genpath('../MatDL'));
%% Generate NARMA Series
% N=300;
% Xin = gen_narma_30(N);
% Xin=Xin(100:end);
% %X = round(rand(1,1000));
% T = [(Xin(2:end)-Xin(1:end-1) > 0)];

ACC=dlmread('HAPT dataset/RawData/acc_exp01_user01.txt');
Xact=ACC(300:315,1:3); %2222:2377
Xact=Xact';
Xact=normalize(Xact);
%X=X(100:end);
%X = round(rand(1,1000));
%T = [(Xin(2:end)-Xin(1:end-1) > 0)];

%pattern_act=return_pattern(Xact);
%pattern=[1 0 1];
%% Get the full motion sequence
Xin=ACC(1:20000,1:3);
%X=ACC(:,1);
Xin=Xin';
Xin=normalize(Xin);
%X=X(100:end);
%X = round(rand(1,1000));
%T = [(Xin(2:end)-Xin(1:end-1) > 0)];
T(1,:)=return_pattern(Xin(1,:));
T(2,:)=return_pattern(Xin(2,:));
T(3,:)=return_pattern(Xin(3,:));
%%
[idxs,pattern]=find_mod_pattern_6D(Xact,T,4,size(Xact,2)*2,10000);
%idxs=strfind(T,pattern);
Tp=zeros(3,size(Xin,2));
Tp(1:3,idxs)=1;

subplot(3,1,1);
plot(1:size(Xin,2),Xin(1,1:end))
hold on;
keep_on=0;
for i=2:size(Xin,2)
    if (Tp(1,i) > 0)
        keep_on=size(pattern,2);
    end
    
    if (keep_on > 0) 
        plot([i-1 i],[Xin(1,i-1) Xin(1,i)],'m-')
        keep_on = keep_on-1;
    end
end

subplot(3,1,2);
plot(1:size(Xin,2),Xin(2,1:end))
hold on;
keep_on=0;
for i=2:size(Xin,2)
    if (Tp(2,i) > 0)
        keep_on=size(pattern,2);
    end
    
    if (keep_on > 0) 
        plot([i-1 i],[Xin(2,i-1) Xin(2,i)],'m-')
        keep_on = keep_on-1;
    end
end

subplot(3,1,3);
plot(1:size(Xin,2),Xin(3,1:end))
hold on;
keep_on=0;
for i=2:size(Xin,2)
    if (Tp(3,i) > 0)
        keep_on=size(pattern,2);
    end
    
    if (keep_on > 0) 
        plot([i-1 i],[Xin(3,i-1) Xin(3,i)],'m-')
        keep_on = keep_on-1;
    end
end
hold off;

X_f=zeros(size(Xin,2)-(size(pattern,2)),size(pattern,2)+1,3);
Y=[];
XVal_f=zeros(size(Xin,2)-(size(pattern,2)),size(pattern,2)+1,3);
YVal=[];
ctr_a=1;
ctr_b=1;
for i=1:size(Xin,2)-(size(pattern,2))
    if mod(i,3) ~= 0
        if (Tp(1,i) == 1)
            X_f(ctr_a,:,1:3)=Xin(1:3,i:i+size(pattern,2))';
            Y=[Y;[1 0]];
        else
            X_f(ctr_a,:,1:3)=Xin(1:3,i:i+size(pattern,2))';
            Y=[Y;[0 1]];
        end
        ctr_a=ctr_a+1;
    else
        if (Tp(1,i) == 1)
            XVal_f(ctr_b,:,1:3)=Xin(1:3,i:i+size(pattern,2))';
            YVal=[YVal;[1 0]];
        else
            XVal_f(ctr_b,:,1:3)=Xin(1:3,i:i+size(pattern,2))';
            YVal=[YVal;[0 1]];
        end
        ctr_b=ctr_b+1;
    end  
end

for idx=1:size(Y,1)
    X(idx,:,1:3)=X_f(idx,:,1:3);
end

for idx=1:size(YVal,1)
    XVal(idx,:,1:3)=XVal_f(idx,:,1:3);
end

%% Load data
% load('../Data/mnist_uint8.mat');
% %X = double(reshape(train_x',28,28,60000))/255;
% X = double(reshape(train_x',14,56,60000))/255;
% X = permute(X, [3 2 1]);
% %XVal = double(reshape(test_x',28,28,10000))/255;
% XVal = double(reshape(test_x',14,56,10000))/255;
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
maxRelError = gradcheck(@two_ctrnn, x, model, y, opt, 2); %last argument should be less or equal to the number of classes

%% Train
[model, trainLoss, trainAccuracy, valLoss, valAccuracy, opt] = train( X, Y, XVal, YVal, model, @two_ctrnn, opt );

%% Predict
[yplabel, confidence, classes, classConfidences, yp] = predict(XVal, @two_ctrnn, model, opt);
%
