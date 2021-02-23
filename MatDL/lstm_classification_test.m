clear;
W_SIZE=0;
load('../Data/mnist_uint8.mat');
X = double(reshape(train_x',784,1,60000))/255; 
X = permute(X, [2 3 1]);
XVal=double(reshape(test_x',784,1,10000))/255; 
XVal = permute(XVal, [2 3 1]);
Y = double(train_y);
YVal = double(test_y);

XTrain={};
XTrain{1}=[];
ctr=1;
for i=1:W_SIZE+1:size(X,2)  
    XTrain{ctr}=squeeze(X(1,i:i+W_SIZE,:));
    XTrain{ctr}=XTrain{ctr}';
    [~,idx]=find(Y(i,:)>0);
    YTrain(ctr)=categorical(idx);
    ctr=ctr+1;
end

numObservations = numel(XTrain);
for i=1:numObservations
    sequence = XTrain{i};
    sequenceLengths(i) = size(sequence,2);
end

% figure
% bar(sequenceLengths)
% ylim([0 30])
% xlabel("Sequence")
% ylabel("Length")
% title("Sorted Data")

[sequenceLengths,idx] = sort(sequenceLengths);
XTrain = XTrain(idx);
YTrain = YTrain(idx)';

inputSize = W_SIZE+1;
numHiddenUnits = 5;
numClasses = 10;

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]

maxEpochs = 100;
miniBatchSize = W_SIZE+1;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,layers,options);
%%
XTest={};
XTest{1}=[];
ctr=1;
for i=1:W_SIZE+1:size(XVal,2)  
    XTest{ctr}=squeeze(XVal(1,i:i+W_SIZE,:));
    XTest{ctr}=XTest{ctr}';
    [~,idx]=find(YVal(i,:)>0);
    YTest(ctr)=categorical(idx);
    ctr=ctr+1;
end

numObservationsTest = numel(XTest);
for i=1:numObservationsTest
    sequence = XTest{i};
    sequenceLengthsTest(i) = size(sequence,2);
end
[sequenceLengthsTest,idx] = sort(sequenceLengthsTest);
XTest = XTest(idx);
YTest = YTest(idx)';

miniBatchSize = W_SIZE+1;
YPred = classify(net,XTest, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');

acc = sum(YPred == YTest)./numel(YTest)