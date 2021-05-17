% A complete example of a continuous time recurrent neural network
%% Init
clear;
addpath(genpath('../MatDL'));
%% Set Params
USE_QUANT=0; 
%AUG_PER_WINDOW=10; %LARGE VALUE WILL INCREASE MEMBERSHIP OF CLASS 1
%The window size should be less than the mean activiy time
PHY_WINDOW_SZ=1;
WINDOW_SZ=round(6.0*50);AUG_PER_WINDOW=3; %# of secs x 50Hz %longer activities generally need longer window sizes ACT 1,2,3 WALKING

%Genetic algorithm mutated synthetic observations flag
AUGMENT_DS=1;
%Extent of difference between the orginal data and the genetically mutated data. Higher value introduces observtaions that may not belong to the original observation distribution 
NCPY_EXT=0.05; % K
%Choose after how many observations do we want to mutate data in a window 
AFT_EVERY=round(WINDOW_SZ/5); % M
%Choose how many levels of quantization should be present. Only 2 and 3 are supported
QUANT_LVLS=3;

%Read all relevant experiment data
ACC1=dlmread('HAPT dataset/RawData/acc_exp01_user01.txt');
ACC2=dlmread('HAPT dataset/RawData/acc_exp02_user01.txt');
ACC4=dlmread('HAPT dataset/RawData/acc_exp04_user02.txt');
ACC5=dlmread('HAPT dataset/RawData/acc_exp05_user03.txt');
ACC7=dlmread('HAPT dataset/RawData/acc_exp07_user04.txt');
ACC9=dlmread('HAPT dataset/RawData/acc_exp09_user05.txt');

%% Get the full motion sequence
Xin1=ACC1(1:20598,1:3); % EXP 1 USR 1
Xin2=ACC2(1:19286,1:3); % EXP 2 USR 1
Xin4=ACC4(1:16565,1:3); % EXP 4 USR 2
Xin5=ACC5(1:20994,1:3); % EXP 5 USR 3
Xin7=ACC7(1:17668,1:3); % EXP 7 USR 4
Xin9=ACC9(1:16864,1:3); % EXP 9 USR 5

%Consolidate the observations
Xin=[Xin1;Xin2;Xin4;Xin5;Xin7;Xin9];
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
Tp4=zeros(3,size(Xin4,1));
Tp5=zeros(3,size(Xin5,1));
Tp7=zeros(3,size(Xin7,1));
Tp9=zeros(3,size(Xin9,1));

% %%1,2,3 WALKING, WALKING UP, WALKING DOWN
%EXP 1 USR 1 ACT 1
Tp1(1:3,7496:round((7496+8078)/1))=1;
Tp1(1:3,8356:round((8356+9250)/1))=1;
Tp1(1:3,9657:round((9657+10567)/1))=1;
Tp1(1:3,10750:round((10750+11714)/1))=1;
% %EXP 1 USR 1 ACT 2
Tp1(1:3,14069:round((14069+14699)/1))=1;
Tp1(1:3,15712:round((15712+16377)/1))=1;
Tp1(1:3,17298:round((17298+17970)/1))=1;
%%EXP 1 USR 1 ACT 3
Tp1(1:3,14069:round((14069+14699)/1))=1;
Tp1(1:3,15712:round((15712+16377)/1))=1;
Tp1(1:3,17298:round((17298+17970)/1))=1;
%%EXP 2 USR 1 ACT 1
Tp2(1:3,7624:round((7624+8252)/1))=1;
Tp2(1:3,8618:round((8618+9576)/1))=1;
Tp2(1:3,9991:round((9991+10927)/1))=1;
Tp2(1:3,11311:round((11311+12282)/2))=1;
%%EXP 2 USR 1 ACT 2
Tp2(1:3,14128:round((14128+14783)/1))=1;
Tp2(1:3,15920:round((15920+16598)/1))=1;
Tp2(1:3,17725:round((17725+18425)/1))=1;
%%EXP 2 USR 1 ACT 3
Tp2(1:3,13129:round((13129+13379)/1))=1;
Tp2(1:3,13495:round((13495+13927)/1))=1;
Tp2(1:3,15037:round((15037+15684)/1))=1;
Tp2(1:3,16847:round((16847+17471)/1))=1;
%EXP 4 USR 2 ACT 1
Tp4(1:3,7306:round((7306+8343)/1))=1; %Mark the beginning 50% of the samples for an activity as 1 and the rest zero
Tp4(1:3,8720:round((8720+9686)/1))=1;
%%EXP 4 USR 2 ACT 2
Tp4(1:3,11294:round((11294+11928)/1))=1; %Mark the beginning 50% of the samples for an activity as 1 and the rest zero
Tp4(1:3,12986:round((12986+13602)/1))=1;
Tp4(1:3,14705:round((14705+15274)/1))=1;
%%EXP 4 USR 2 ACT 3
Tp4(1:3,10438:round((10438+11056)/1))=1; %Mark the beginning 50% of the samples for an activity as 1 and the rest zero
Tp4(1:3,12171:round((12171+12732)/1))=1;
Tp4(1:3,13862:round((13862+14444)/1))=1;
%%EXP 5 USR 3 ACT 1
Tp5(1:3,8687:round((8687+9837)/1))=1;
Tp5(1:3,10240:round((10240+11305)/1))=1;
%%EXP 5 USR 3 ACT 2
Tp5(1:3,14018:round((14018+14694)/1))=1;
Tp5(1:3,15985:round((15985+16611)/1))=1;
Tp5(1:3,17811:round((17811+18477)/1))=1;
Tp5(1:3,19536:round((19536+20152)/1))=1;
%%EXP 5 USR 3 ACT 3
Tp5(1:3,15098:round((15098+15693)/1))=1;
Tp5(1:3,16964:round((16964+17559)/1))=1;
%%EXP 7 USR 4 ACT 1
Tp7(1:3,8125:round((8125+9269)/1))=1;
Tp7(1:3,9443:round((9443+10483)/1))=1;
%%EXP 7 USR 4 ACT 2
Tp7(1:3,12653:round((12653+13437)/1))=1;
Tp7(1:3,14548:round((14548+15230)/1))=1;
Tp7(1:3,16178:round((16178+16814)/1))=1;
%%EXP 7 USR 4 ACT 3
Tp7(1:3,11340:round((11340+11596)/1))=1;
Tp7(1:3,11824:round((11824+12047)/1))=1;
Tp7(1:3,13625:round((13625+14341)/1))=1;
Tp7(1:3,15391:round((15391+16025)/1))=1;
%%EXP 9 USR 5 ACT 1
Tp9(1:3,7891:round((7891+9023)/1))=1;
Tp9(1:3,9216:round((9216+10237)/1))=1;
%%EXP 9 USR 5 ACT 2
Tp9(1:3,11867:round((11867+12553)/1))=1;
Tp9(1:3,13567:round((13567+14201)/1))=1;
Tp9(1:3,15087:round((15087+15725)/1))=1;
%%EXP 9 USR 5 ACT 3
Tp9(1:3,11051:round((11051+11704)/1))=1;
Tp9(1:3,12795:round((12795+13414)/1))=1;
Tp9(1:3,14384:round((14384+14963)/1))=1;

%The ground truth label  vector
Tp=[Tp1 Tp2 Tp4 Tp5 Tp7 Tp9];

%Create Training and Test Datasets
X_f=zeros(ceil((size(Xin,2)-WINDOW_SZ)/WINDOW_SZ),WINDOW_SZ,2);
Y_f=[];
XVal_f=zeros(ceil((size(Xin,2)-WINDOW_SZ)/WINDOW_SZ),WINDOW_SZ,2);
YVal_f=[];
ctr_a=1;
ctr_b=1;
counter=1;
for i=1:WINDOW_SZ:size(Xin,2)-WINDOW_SZ
    if mod(counter,3) ~= 0
        %if (sum(Tp(1,i:i+WINDOW_SZ)) >= WINDOW_SZ/2.0)
        if (sum(Tp(1,i:i+WINDOW_SZ)) >= WINDOW_SZ/3.0)
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
        if (sum(Tp(1,i:i+WINDOW_SZ)) >= WINDOW_SZ/2.0)
        %if (sum(Tp(1,i:i+WINDOW_SZ)) >= WINDOW_SZ/3.0)
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
tot1=0;
tot2=0;
last_idx=0;
X=zeros(size(Y_f,1),size(X_f,2),size(X_f,3));
max_card=max(sum(Y_f(:,1)),sum(Y_f(:,2)));
Y=[];
for idx=1:size(Y_f,1)
    if (Y_f(idx,2) == 1) && (tot2 < max_card)
        tot2=tot2+1;
        last_idx=last_idx+1;
        X(last_idx,:,1:3)=X_f(idx,:,1:3);
        Y=[Y;[0 1]];
    end
    
    if (Y_f(idx,1) == 1) && (tot1 < max_card)
        tot1=tot1+1;
        last_idx=last_idx+1;
        X(last_idx,:,1:3)=X_f(idx,:,1:3);
        Y=[Y;[1 0]];
    end
    
    if ((tot1 >= max_card) || (tot2 >= max_card))
        break;
    end
end
Y=Y(1:last_idx,:);
X=X(1:last_idx,:,:);

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
X = permute(X, [1 3 2]);
XVal=permute(XVal, [1 3 2]);
opt = struct;
%layers_size=[5 5];
layers_size=[3];
TS=0.01;
[model, opt] = init_two_ctrnnm(3, 2, layers_size, opt);

%% Set Hyper-parameters
opt.batchSize = 30;
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