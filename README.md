# CTRNN based Simulation for MEMS based Neural Networks

[![status](https://cdn.ncbi.nlm.nih.gov/corehtml/query/egifs/https:--res.mdpi.com-data-pubmed-linkout.png)](https://www.mdpi.com/2072-666X/12/3/268)
[![DOI](https://zenodo.org/badge/103798084.svg)](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2021.731076/full)

This simulation based on *MatDL*, an open-source lightweight deep learning library native in MATLAB that implements some most commonly used deep learning algorithms. 
CTRNN simulation for MEMS based Neural Networks offer a deiciated backpropagation through time algorithm for recurrent neural networks (RNNs). Several boilerplate functions for training, gradients checking, etc. are also implemented for this simulation.

## Installation

- Add *MatDL* to your path:
```matlab
addpath(genpath('MatDL'));
```

- Compile the C MEX files using the `Makefile` or using the following:
```matlab
cd MatDL/convnet; mex col2im_mex.c; mex im2col_mex.c; cd ..; cd ..;
```

## Project Layout

`common/`: Basic building blocks of neural networks such as nonlinear functions, etc.

`convnet/`: ConvNet specific functions such as convolutional layers, max pooling layers, etc.

`nn/`: NN specific functions.

`optim/`: Optimization algorithms such as SGD, RMSProp, ADAM, etc.

`rnn/`: RNN and LSTM functions.

`train/`: Functions for gradients checking, training, and prediction.

`zoo/`: Samples of various models definitions and initializations.

## Usage

This is a sample complete minimum working example:
(Examples: [DNN](MatDL/nnet.m) (Below), [ConvNet](MatDL/convnet.m), [RNN](MatDL/rnnet.m))

```matlab
% A complete minimum working example.
%% Init
clear all
addpath(genpath('../MatDL'));

%% Load data
load('../Data/mnist_uint8.mat'); % Replace with your data file
X = double(train_x)/255; Y = double(train_y);
XVal = double(test_x)/255; YVal =  double(test_y);

rng(0);

%% Initialize model
opt = struct;
[model, opt] = init_six_nn_bn(784, 10, [100, 100, 100, 100, 100], opt);

%% Hyper-parameters
opt.batchSize = 100;

opt.optim = @rmsprop;
% opt.beta1 = 0.9; opt.beta2 = 0.999; opt.t = 0; opt.mgrads = opt.vgrads;
opt.rmspropDecay = 0.99;
% opt.initialMomentum = 0.5; opt.switchEpochMomentum = 1; opt.finalMomentum = 0.9;
opt.learningRate = 0.01;
opt.learningDecaySchedule = 'stepsave'; % 'no_decay', 't/T', 'step'
opt.learningDecayRate = 0.5;
opt.learningDecayRateStep = 5;

opt.dropout = 0.5;
opt.weightDecay = false;
opt.maxNorm = false;

opt.maxEpochs = 100;
opt.earlyStoppingPatience = 20;
opt.valFreq = 100;

opt.plotProgress = true;
opt.extractFeature = false;
opt.computeDX = false;

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
x = X(1:100,:);
y = Y(1:100,:);
maxRelError = gradcheck(@six_nn_bn, x, model, y, opt, 10);

%% Train
[model, trainLoss, trainAccuracy, valLoss, valAccuracy, opt] = train(X, Y, XVal, YVal, model, @six_nn_bn, opt);

%% Predict
[yplabel, confidence, classes, classConfidences, yp] = predict(XVal, @six_nn_bn, model, opt)
```

## CTRNN and MEMS-CTRNN Model Files

- As a convention, any file that is particularly related to CTRNN model is preceded by 'ctrnn_' in its name. The same rule applies to MEMS CTRNN model where relevant files are preceded by ctrnnm_.

- An example of code that uses CTRNN model is located at CTRNN_timeseries/MatDL/ctrnnet_UCI_seq_6D_walkingup_tuned.m
- An example of code that uses MEMS CTRNN model is located at CTRNN_timeseries/MatDL/ctrnnm_UCI_seq_6D_walkingup_tuned.m

MEMS CTRNN is still under development.

Important files that are relevant to MEMS CTRNN implementation
- 'CTRNN_timeseries/MatDL/rnn/ctrnnm_forward.m': The MEMS-CTRNN forward pass equation is implemented here. Voltages asscoiated to each neuron are passed to this function but are not used in the training process. 
- 'CTRNN_timeseries/MatDL/zoo/init_two_ctrnnm.m': We initialize all the parameters required for training for MEMS-CTRNN here
- 'CTRNN_timeseries/MatDL/zoo/two_ctrnnm.m': Training functions are called in this module 
- 'CTRNN_timeseries/MatDL/optim': You can change the training optimizer files here. Make sure to exclude a parameter in these files, if you do not want to use it in the training process.

## License

MIT
