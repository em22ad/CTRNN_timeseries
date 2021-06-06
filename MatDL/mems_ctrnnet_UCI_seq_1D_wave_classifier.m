% A complete example of a continuous time recurrent neural network
%% Init
clear;
addpath(genpath('../MatDL'));
%% Set Params
USE_QUANT=0; 
%AUG_PER_WINDOW=10; %LARGE VALUE WILL INCREASE MEMBERSHIP OF CLASS 1
%The window size should be less than the mean activiy time
PHY_WINDOW_SZ=1;
WINDOW_SZ=10;AUG_PER_WINDOW=10; %# of secs x 50Hz %longer activities generally need longer window sizes ACT 7 STAND_TO_SIT

%Genetic algorithm mutated synthetic observations flag
AUGMENT_DS=0;
%Extent of difference between the orginal data and the genetically mutated data. Higher value introduces observtaions that may not belong to the original observation distribution 
NCPY_EXT=0.05; % K
%Choose after how many observations do we want to mutate data in a window 
AFT_EVERY=round(WINDOW_SZ/5); % M
%Choose how many levels of quantization should be present. Only 2 and 3 are supported
QUANT_LVLS=3;
%Choose among the dimensions 1(X), 2(Y), 3(Z) that you want to use
DIMS=[1];%[1 3];
%% Get the square and sawtooth wave
Fs = 1500;                  % Sampling frequency 1500 Hz 
t = 0:1/Fs:4;%.1;               % 0 to .1 sec duration
f = 60;                     % Frequency of signal 60 Hz

SW(:,1) = square(2*pi*f*t);
SW(:,2) = square(2*pi*f/2*t);
SW(:,3) = square(2*pi*f/3*t);
Xin1=SW; % Square waves at multiple frequencies (important for future tuning of /rho)

STW(:,1) = sawtooth(2*pi*f*t,1/2);
STW(:,2) = sawtooth(2*pi*f/2*t,1/2);
STW(:,3) = sawtooth(2*pi*f/3*t,1/2);
Xin2=STW; % Triangle waves at multiple frequencies

%Consolidate the observations
Xin=[Xin1;Xin2];
Xin=Xin';

%Normalize the observations for each sensor
Xin=normalize(Xin,2);

%construct the ground truth
if (QUANT_LVLS==2)
    T(1,:)=return_pattern(Xin(1,:));
    T(2,:)=return_pattern(Xin(2,:));
    T(3,:)=return_pattern(Xin(3,:));
elseif (QUANT_LVLS==3)
    T(1,:)=return_pattern_3(Xin(1,:));
    T(2,:)=return_pattern_3(Xin(2,:));
    T(3,:)=return_pattern_3(Xin(3,:));
end
%%

Tp1=zeros(3,size(Xin1,1));
Tp2=zeros(3,size(Xin2,1));

%%Ground Truth
%Square
Tp1(1:3,1:size(Xin1,1))=1;
% %Triangle
% Tp2(1:3,146:302)=0;

%The ground truth label  vector
Tp=[Tp1 Tp2];

% plot(Tp(1,:))
% hold on;
% plot(Xin(1,:))
% hold off;

%Create Training and Test Datasets
mod_div=3;
sz_x=((size(Xin,2)-WINDOW_SZ)/WINDOW_SZ);
sz_x=(sz_x-sz_x/mod_div);
X_f=zeros(ceil(sz_x),WINDOW_SZ,2);
Y_f=[];
XVal_f=zeros(ceil(sz_x),WINDOW_SZ,2);
YVal_f=[];
ctr_a=1;
ctr_b=1;
counter=1;
for i=1:WINDOW_SZ:size(Xin,2)-WINDOW_SZ
    if mod(counter,mod_div) ~= 0
        %if (sum(Tp(1,i:i+WINDOW_SZ)) >= WINDOW_SZ/1.33)
        if (sum(Tp(1,i:i+WINDOW_SZ-1)) >= WINDOW_SZ/2.0)
            if (USE_QUANT == 0)
                X_f(ctr_a,:,1:3)=Xin(1:3,i:i+WINDOW_SZ-1)';
            else
                X_f(ctr_a,:,1:3)=T(1:3,i:i+WINDOW_SZ-1)';
            end
            Y_f=[Y_f;[1 0]];
            ctr_a=ctr_a+1;
            
            if (AUGMENT_DS == 1)
                ii=1;
                while (ii <= AUG_PER_WINDOW)
                    if (USE_QUANT == 0)
                        Mut_X1=gen_variations(Xin(1,i:i+WINDOW_SZ-1),AFT_EVERY,NCPY_EXT);
                        Mut_X2=gen_variations(Xin(2,i:i+WINDOW_SZ-1),AFT_EVERY,NCPY_EXT);
                        Mut_X3=gen_variations(Xin(3,i:i+WINDOW_SZ-1),AFT_EVERY,NCPY_EXT);
                    else
                        Mut_X1=gen_variations(T(1,i:i+WINDOW_SZ-1),AFT_EVERY,NCPY_EXT);
                        Mut_X2=gen_variations(T(2,i:i+WINDOW_SZ-1),AFT_EVERY,NCPY_EXT);
                        Mut_X3=gen_variations(T(3,i:i+WINDOW_SZ-1),AFT_EVERY,NCPY_EXT);
                    end
                    Mut_X=[Mut_X1' Mut_X2' Mut_X3'];
                    X_f(ctr_a,:,1:3)=Mut_X;
                    Y_f=[Y_f;[1 0]];
                    ii=ii+1;
                    ctr_a=ctr_a+1;
                end
            end
        else
            if (USE_QUANT == 0)
                X_f(ctr_a,:,1:3)=Xin(1:3,i:i+WINDOW_SZ-1)';
            else
                X_f(ctr_a,:,1:3)=T(1:3,i:i+WINDOW_SZ-1)';
            end
            Y_f=[Y_f;[0 1]];
            ctr_a=ctr_a+1;
            if (0)%(AUGMENT_DS == 1)
                ii=1;
                while (ii <= AUG_PER_WINDOW)
                    if (USE_QUANT == 0)
                        Mut_X1=gen_variations(Xin(1,i:i+WINDOW_SZ-1),AFT_EVERY,NCPY_EXT);
                        Mut_X2=gen_variations(Xin(2,i:i+WINDOW_SZ-1),AFT_EVERY,NCPY_EXT);
                        Mut_X3=gen_variations(Xin(3,i:i+WINDOW_SZ-1),AFT_EVERY,NCPY_EXT);
                    else
                        Mut_X1=gen_variations(T(1,i:i+WINDOW_SZ-1),AFT_EVERY,NCPY_EXT);
                        Mut_X2=gen_variations(T(2,i:i+WINDOW_SZ-1),AFT_EVERY,NCPY_EXT);
                        Mut_X3=gen_variations(T(3,i:i+WINDOW_SZ-1),AFT_EVERY,NCPY_EXT);
                    end
                    Mut_X=[Mut_X1' Mut_X2' Mut_X3'];
                    X_f(ctr_a,:,1:3)=Mut_X;
                    Y_f=[Y_f;[0 1]];
                    ii=ii+1;
                    ctr_a=ctr_a+1;
                end
            end

        end
    else
        %if (sum(Tp(1,i:i+WINDOW_SZ)) >= WINDOW_SZ/1.33)
        if (sum(Tp(1,i:i+WINDOW_SZ-1)) >= WINDOW_SZ/2.0)
            if (USE_QUANT == 0)
                XVal_f(ctr_b,:,1:3)=Xin(1:3,i:i+WINDOW_SZ-1)';
            else
                XVal_f(ctr_b,:,1:3)=T(1:3,i:i+WINDOW_SZ-1)';
            end
            YVal_f=[YVal_f;[1 0]];        
            ctr_b=ctr_b+1;
            
            if (AUGMENT_DS == 1)
                ii=1;
                while (ii <= AUG_PER_WINDOW)
                    if (USE_QUANT == 0)
                        Mut_X1=gen_variations(Xin(1,i:i+WINDOW_SZ-1),AFT_EVERY,NCPY_EXT);
                        Mut_X2=gen_variations(Xin(2,i:i+WINDOW_SZ-1),AFT_EVERY,NCPY_EXT);
                        Mut_X3=gen_variations(Xin(3,i:i+WINDOW_SZ-1),AFT_EVERY,NCPY_EXT);
                    else
                        Mut_X1=gen_variations(T(1,i:i+WINDOW_SZ-1),AFT_EVERY,NCPY_EXT);
                        Mut_X2=gen_variations(T(2,i:i+WINDOW_SZ-1),AFT_EVERY,NCPY_EXT);
                        Mut_X3=gen_variations(T(3,i:i+WINDOW_SZ-1),AFT_EVERY,NCPY_EXT);
                    end
                    Mut_X=[Mut_X1' Mut_X2' Mut_X3'];
                    XVal_f(ctr_b,:,1:3)=Mut_X;
                    YVal_f=[YVal_f;[1 0]];
                    ii=ii+1;
                    ctr_b=ctr_b+1;
                end
            end
        else
            if (USE_QUANT == 0)
                XVal_f(ctr_b,:,1:3)=Xin(1:3,i:i+WINDOW_SZ-1)';
            else
                XVal_f(ctr_b,:,1:3)=T(1:3,i:i+WINDOW_SZ-1)';
            end
            YVal_f=[YVal_f;[0 1]];
            ctr_b=ctr_b+1;
            if (0)%(AUGMENT_DS == 1)
                ii=1;
                while (ii <= AUG_PER_WINDOW)
                    if (USE_QUANT == 0)
                        Mut_X1=gen_variations(Xin(1,i:i+WINDOW_SZ-1),AFT_EVERY,NCPY_EXT);
                        Mut_X2=gen_variations(Xin(2,i:i+WINDOW_SZ-1),AFT_EVERY,NCPY_EXT);
                        Mut_X3=gen_variations(Xin(3,i:i+WINDOW_SZ-1),AFT_EVERY,NCPY_EXT);
                    else
                        Mut_X1=gen_variations(T(1,i:i+WINDOW_SZ-1),AFT_EVERY,NCPY_EXT);
                        Mut_X2=gen_variations(T(2,i:i+WINDOW_SZ-1),AFT_EVERY,NCPY_EXT);
                        Mut_X3=gen_variations(T(3,i:i+WINDOW_SZ-1),AFT_EVERY,NCPY_EXT);
                    end
                    Mut_X=[Mut_X1' Mut_X2' Mut_X3'];
                    X_f(ctr_b,:,1:3)=Mut_X;
                    Y_f=[Y_f;[0 1]];
                    ii=ii+1;
                    ctr_b=ctr_b+1;
                end
            end
        end
    end  
    counter=counter+1;
end

XVal=zeros(size(YVal_f,1),size(XVal_f,2),size(XVal_f,3));
for idx=1:size(YVal_f,1)
    XVal(idx,:,1:3)=XVal_f(idx,:,1:3);
end

%Make the Training set have equal number of classes
% tot1=0;
% tot2=0;
% last_idx=0;
% X=zeros(size(Y_f,1),size(X_f,2),size(X_f,3));
% max_card=max(sum(Y_f(:,1)),sum(Y_f(:,2)));
% Y=[];
% for idx=1:size(Y_f,1)
%     if (Y_f(idx,2) == 1) && (tot2 < max_card)
%         tot2=tot2+1;
%         last_idx=last_idx+1;
%         X(last_idx,:,1:3)=X_f(idx,:,1:3);
%         Y=[Y;[0 1]];
%     end
%     
%     if (Y_f(idx,1) == 1) && (tot1 < max_card)
%         tot1=tot1+1;
%         last_idx=last_idx+1;
%         X(last_idx,:,1:3)=X_f(idx,:,1:3);
%         Y=[Y;[1 0]];
%     end
%     
%     if ((tot1 >= max_card) || (tot2 >= max_card))
%         break;
%     end
% end
% Y=Y(1:last_idx,:);
% X=X(1:last_idx,:,:);
Y=Y_f;
X=X_f;

%Make the validation set have equal number of classes
tot1=0;
tot2=0;
last_idx=0;
XVal=zeros(size(YVal_f,1),size(XVal_f,2),size(XVal_f,3));
min_card=min(sum(YVal_f(:,1)),sum(YVal_f(:,2)));
YVal=[];
for idx=1:size(YVal_f,1)
    if (YVal_f(idx,2) == 1) && (tot2 < min_card)
        tot2=tot2+1;
        last_idx=last_idx+1;
        XVal(last_idx,:,1:3)=XVal_f(idx,:,1:3);
        YVal=[YVal;[0 1]];
    end
    
    if (YVal_f(idx,1) == 1) && (tot1 < min_card)
        tot1=tot1+1;
        last_idx=last_idx+1;
        XVal(last_idx,:,1:3)=XVal_f(idx,:,1:3);
        YVal=[YVal;[1 0]];
    end
    
    if ((tot1 >= min_card) && (tot2 >= min_card))
        break;
    end
end
YVal=YVal(1:last_idx,:);
XVal=XVal(1:last_idx,:,:);

%% Initialize model
X=X(:,:,DIMS);
XVal=XVal(:,:,DIMS);
X = permute(X, [1 3 2]);
XVal=permute(XVal, [1 3 2]);
opt = struct;
%layers_size=[5 5];
layers_size=[6];
TS=0.01;
[model, opt] = init_two_ctrnnm(size(DIMS,2), 2, layers_size, opt);

%% Set Hyper-parameters
opt.batchSize = 400;
opt.optim = @rmsprop;
% opt.beta1 = 0.9; opt.beta2 = 0.999; opt.t = 0; opt.mgrads = opt.vgrads;
opt.rmspropDecay = 0.9667;%0.75;
% opt.initialMomentum = 0.5; opt.switchEpochMomentum = 1; opt.finalMomentum = 0.9;
opt.learningRate = 0.015;%0.015;
opt.learningDecaySchedule = 'exp'; %'no_decay';%'t/T';%'step';
opt.learningDecayRate = 1;
%opt.learningDecayRateStep = 5;

opt.dropout = 1;
opt.weightDecay = false;
opt.maxNorm = false;

opt.maxEpochs = 1000;
opt.earlyStoppingPatience = 1000;
opt.valFreq = 25;

opt.plotProgress = true;
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
x = X(1:floor(size(X,1)/2), :, :);
y = Y(1:floor(size(Y,1)/2), :, :);
maxRelError = gradcheck(@two_ctrnnm, x, model, y, opt, 2); %last argument should be less or equal to the number of classes

%% Train
[model, trainLoss, trainAccuracy, valLoss, valAccuracy, opt] = train( X, Y, XVal, YVal, model, @two_ctrnnm, opt );

%% Predict
[yplabel, confidence, classes, classConfidences, yp] = predict(XVal, @two_ctrnnm, model, opt);

ylabel=[];
for i=1:size(YVal,1)
    if (YVal(i,1)==1)
        ylabel=[ylabel;1];
    else
        ylabel=[ylabel;2];
    end
end

ypr = categorical(yplabel);
ytest= categorical(ylabel);
plotconfusion(ytest,ypr)