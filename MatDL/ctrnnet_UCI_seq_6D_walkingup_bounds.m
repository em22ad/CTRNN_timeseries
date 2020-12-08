% A complete example of a continuous time recurrent neural network
%% Init
clear all;
addpath(genpath('../MatDL'));
%% Set Params
USE_QUANT=1; 
AUG_PER_WINDOW=10;
WINDOW_SZ=round(3.49*50); %# of secs x 50Hz
AUGMENT_DS=0;
NCPY_EXT=0.05; % K
AFT_EVERY=round(WINDOW_SZ/15);%round(WINDOW_SZ/5); % M
QUANT_LVLS=3;
DISPLAY_SAMPLES=0;
DISPLAY_3D_OBS=1;
SHOW_DEC_BOUND=0;
SHOW_DEC_BOUND_CLASSIFIER=1;

% B=Y(Y(:,1)>0)
% size(B)
% size(Y)

% N=300;
% Xin = gen_narma_30(N);
% Xin=Xin(100:end);
% %X = round(rand(1,1000));
% T = [(Xin(2:end)-Xin(1:end-1) > 0)];

%ACC=dlmread('HAPT dataset/RawData/acc_exp01_user01.txt');
%ACC=dlmread('HAPT dataset/RawData/acc_exp04_user02.txt');
ACC=dlmread('HAPT dataset/RawData/acc_exp05_user03.txt');
%ACC=dlmread('HAPT dataset/RawData/acc_exp07_user04.txt');
%ACC=dlmread('HAPT dataset/RawData/acc_exp09_user05.txt');

% Xact=ACC(300:315,1:3); %2222:2377
% Xact=Xact';
% Xact=normalize(Xact);

%X=X(100:end);
%X = round(rand(1,1000));
%T = [(Xin(2:end)-Xin(1:end-1) > 0)];

%pattern_act=return_pattern(Xact);
%pattern=[1 0 1];
%% Get the full motion sequence
%Xin=ACC(1:20000,1:3);
%Xin=ACC(1:16565,1:3); % EXP 4 USR 2
Xin=ACC(1:20994,1:3); % EXP 5 USR 3
%Xin=ACC(1:17668,1:3); % EXP 7 USR 4
%Xin=ACC(1:16864,1:3);

%X=ACC(:,1);
xx=normalize(Xin(:,1));
yy=normalize(Xin(:,2));
zz=normalize(Xin(:,3));
Xin=[xx yy zz];
%Xin=normalize(Xin);
Xin=Xin';
%X=X(100:end);
%X = round(rand(1,1000));
%T = [(Xin(2:end)-Xin(1:end-1) > 0)];
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
%[idxs,pattern]=find_mod_pattern_6D(Xact,T,4,size(Xact,2)*2,10000);
Tp=zeros(3,size(Xin,2));
T1=zeros(3,size(Xin,2));
T7=zeros(3,size(Xin,2));
T8=zeros(3,size(Xin,2));
T9=zeros(3,size(Xin,2));
T10=zeros(3,size(Xin,2));
T11=zeros(3,size(Xin,2));
T12=zeros(3,size(Xin,2));
%EXP 1 USR 1 ACT 2
%Tp(1:3,14069:round((14069+14699)/2))=1;
%Tp(1:3,15712:round((15712+16377)/2))=1;
%Tp(1:3,17298:round((17298+17970)/2))=1;
% %EXP 4 USR 2 ACT 2
%Tp(1:3,11294:round((11294+11928)/1.5))=1; %Mark the beginning 50% of the samples for an activity as 1 and the rest zero
%Tp(1:3,12986:round((12986+13602)/1.5))=1;
%Tp(1:3,14705:round((14705+15274)/1.5))=1;

%EXP 5 USR 3 ACT 2 (WALK UPSTAIRS)
Tp(1:3,14018:round((14018+14694)/2))=1;
Tp(1:3,15985:round((15985+16611)/2))=1;
Tp(1:3,17811:round((17811+18477)/2))=1;
Tp(1:3,19536:round((19536+20152)/2))=1;
%EXP 5 USR 3 ACT 1 (WALKING)
Tp(1:3,8687:round((8687+9837)/2))=1;
Tp(1:3,10240:round((10240+11305)/2))=1;
%EXP 5 USR 3 ACT 3 (WALKING DOWNSTAIRS)
Tp(1:3,15098:round((15098+15693)/2))=1;
Tp(1:3,16964:round((16964+17559)/2))=1;
Tp(1:3,18810:round((18810+19325)/2))=1;


%EXP 5 USR 3 ACT 2 (WALK UPSTAIRS)
T1(1:3,14018:14694)=1;
T1(1:3,15985:16611)=1;
T1(1:3,17811:18477)=1;
T1(1:3,19536:20152)=1;
%EXP 5 USR 3 ACT 1 (WALKING)
T1(1:3,8687:9837)=1;
T1(1:3,10240:11305)=1;
%EXP 5 USR 3 ACT 3 (WALKING DOWNSTAIRS)
T1(1:3,15098:15693)=1;
T1(1:3,16964:17559)=1;
T1(1:3,18810:19325)=1;
%EXP 5 USR 3 ACT 7 (STAND_TO_SIT)
T7(1:3,1365:1506)=1;
%EXP 5 USR 3 ACT 8 (SIT_TO_STAND)
T8(1:3,2361:2470)=1;
%EXP 5 USR 3 ACT 9 (SIT_TO_LIE)
T9(1:3,6060:6210)=1;
%EXP 5 USR 3 ACT 10 (LIE_TO_SIT)
T10(1:3,4851:5074)=1;
%EXP 5 USR 3 ACT 11 (STAND_TO_LIE) 
T11(1:3,3619:3834)=1;
%EXP 5 USR 3 ACT 12 (LIE_TO_STAND) 
T12(1:3,7379:7562)=1;

%EXP 7 USR 4 ACT 2
% Tp(1:3,12653:round((12653+13437)/2))=1;
% Tp(1:3,14548:round((14548+15230)/2))=1;
% Tp(1:3,16178:round((16178+16814)/2))=1;
%EXP 9 USR 5 ACT 2
% Tp(1:3,11867:round((11867+12553)/2))=1;
% Tp(1:3,13567:round((13567+14201)/2))=1;
% Tp(1:3,15087:round((15087+15725)/2))=1;

%We use non-overlapping windows
mark_sz=1;

    HXVal_f=zeros(size(Xin,2)-WINDOW_SZ,WINDOW_SZ+1,3);
    HYVal=[];
    Hctr_b=1;
    for i=1:WINDOW_SZ:size(Xin,2)-WINDOW_SZ
            HXVal_f(Hctr_b,:,1:3)=Xin(1:3,i:i+WINDOW_SZ)';
            HYVal=[HYVal;[1 0]]; %TBD: See if we need to label this data tp begin with       
            
            if (SHOW_DEC_BOUND == 1)
                hold on;
                if (Tp(1,i)-round(rand()) > 0)
                    plot3(HXVal_f(Hctr_b,:,1),HXVal_f(Hctr_b,:,2),HXVal_f(Hctr_b,:,3),'r.','MarkerSize',mark_sz+8)
                else
                    plot3(HXVal_f(Hctr_b,:,1),HXVal_f(Hctr_b,:,2),HXVal_f(Hctr_b,:,3),'b.','MarkerSize',mark_sz)
                end
            end
            rec_idx=Hctr_b;
            Hctr_b=Hctr_b+1;
            ii=1;
            while (ii <= AUG_PER_WINDOW)
                Mut_X1=gen_variations(Xin(1,i:i+WINDOW_SZ),AFT_EVERY,NCPY_EXT);
                Mut_X2=gen_variations(Xin(2,i:i+WINDOW_SZ),AFT_EVERY,NCPY_EXT);
                Mut_X3=gen_variations(Xin(3,i:i+WINDOW_SZ),AFT_EVERY,NCPY_EXT);

                Mut_X=[Mut_X1' Mut_X2' Mut_X3'];
                HXVal_f(Hctr_b,:,1:3)=Mut_X;
                HYVal=[HYVal;[1 0]]; %TBD: See how and if to label
                ii=ii+1;
                if (SHOW_DEC_BOUND == 1)
                    hold on;
                    xx=norm(HXVal_f(Hctr_b,:,1)-HXVal_f(rec_idx,:,1));
                    yy=norm(HXVal_f(Hctr_b,:,1)-HXVal_f(rec_idx,:,1));
                    zz=norm(HXVal_f(Hctr_b,:,1)-HXVal_f(rec_idx,:,1));
                    tot=sqrt(xx^2+yy^2+zz^2);
                    if (Tp(1,i) > 0) && (tot < 0.1)
                        plot3(HXVal_f(Hctr_b,:,1),HXVal_f(Hctr_b,:,2),HXVal_f(Hctr_b,:,3),'r.','MarkerSize',mark_sz+9)
                    elseif (Tp(1,i) > 0) 
                        plot3(HXVal_f(Hctr_b,:,1),HXVal_f(Hctr_b,:,2),HXVal_f(Hctr_b,:,3),'g.','MarkerSize',mark_sz+5)
                    else
                        plot3(HXVal_f(Hctr_b,:,1),HXVal_f(Hctr_b,:,2),HXVal_f(Hctr_b,:,3),'b.','MarkerSize',mark_sz)
                    end
                end
                Hctr_b=Hctr_b+1;
            end
    end        
    
    if (SHOW_DEC_BOUND == 1)
        hold off;
    end
    
    for idx=1:size(HYVal,1)
        HXVal(idx,:,1:3)=HXVal_f(idx,:,1:3);
    end
        


if (DISPLAY_3D_OBS == 1)
    figure;
    hold on;
    %plot3(Xin(1,1),Xin(2,1),Xin(3,1),'b.')

    %%xlim([min(Xin(1,:)) max(Xin(1,:))])
    %%ylim([min(Xin(2,:)) max(Xin(2,:))])
    %%zlim([min(Xin(3,:)) max(Xin(3,:))])
    
    %%for i=2:size(Xin,2)
        %plot3(Xin(1,2:end),Xin(2,2:end),Xin(3,2:end),'b.')
        %plot3(ACC(1:20994,1),ACC(1:20994,2),ACC(1:20994,3),'b.')
        
    %%end
    keep_on_1=0;
    keep_on_7=0;
    keep_on_8=0;
    keep_on_9=0;
    keep_on_10=0;
    keep_on_11=0;
    keep_on_12=0;
    for i=2:size(Xin,2)
        if (T1(1,i) > 0)
            keep_on_1=WINDOW_SZ;
        end
        if (T7(1,i) > 0)
            keep_on_7=WINDOW_SZ;
        end
        if (T8(1,i) > 0)
            keep_on_8=WINDOW_SZ;
        end
        if (T9(1,i) > 0)
            keep_on_9=WINDOW_SZ;
        end
        if (T10(1,i) > 0)
            keep_on_10=WINDOW_SZ;
        end
        if (T11(1,i) > 0)
            keep_on_11=WINDOW_SZ;
        end
        if (T12(1,i) > 0)
            keep_on_12=WINDOW_SZ;
        end
        

        if (keep_on_1 > 0) 
            h1=plot3(Xin(1,i),Xin(2,i),Xin(3,i),'m.','DisplayName','walk(incl up & down)');
            %plot3(ACC(i,1),ACC(i,2),ACC(i,3),'mo')
            keep_on_1 = keep_on_1-1;
        elseif (keep_on_7 > 0)
            h7=plot3(Xin(1,i),Xin(2,i),Xin(3,i),'r+','DisplayName','stand2sit');
            keep_on_7 = keep_on_7-1;
        elseif (keep_on_8 > 0)
            h8=plot3(Xin(1,i),Xin(2,i),Xin(3,i),'ro','DisplayName','sit2stand');
            keep_on_8 = keep_on_8-1;
        elseif (keep_on_9 > 0)
            h9=plot3(Xin(1,i),Xin(2,i),Xin(3,i),'g+','DisplayName','sit2lie');
            keep_on_9 = keep_on_9-1;
        elseif (keep_on_10 > 0)
            h10=plot3(Xin(1,i),Xin(2,i),Xin(3,i),'go','DisplayName','lie2sit');
            keep_on_10 = keep_on_10-1;
        elseif (keep_on_11 > 0)
            h11=plot3(Xin(1,i),Xin(2,i),Xin(3,i),'k+','DisplayName','stand2lie');
            keep_on_11 = keep_on_11-1;
        elseif (keep_on_12 > 0)
            h12=plot3(Xin(1,i),Xin(2,i),Xin(3,i),'ko','DisplayName','lie2stand');
            keep_on_12 = keep_on_12-1;
        else
            h13=plot3(Xin(1,i),Xin(2,i),Xin(3,i),'b.','DisplayName','others');
        end
    end
    hold off;
    legend([h1 h7 h8 h9 h10 h11 h12 h13])
end


if (DISPLAY_SAMPLES == 1)
    subplot(3,1,1);
    plot(1:size(Xin,2),Xin(1,1:end))
    hold on;
    keep_on=0;
    for i=2:size(Xin,2)
        if (Tp(1,i) > 0)
            keep_on=WINDOW_SZ;
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
            keep_on=WINDOW_SZ;
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
            keep_on=WINDOW_SZ;
        end

        if (keep_on > 0) 
            plot([i-1 i],[Xin(3,i-1) Xin(3,i)],'m-')
            keep_on = keep_on-1;
        end
    end
    hold off;
end

%%
X_f=zeros(size(Xin,2)-WINDOW_SZ,WINDOW_SZ+1,3);
Y=[];
XVal_f=zeros(size(Xin,2)-WINDOW_SZ,WINDOW_SZ+1,3);
YVal=[];
ctr_a=1;
ctr_b=1;
for i=1:WINDOW_SZ:size(Xin,2)-WINDOW_SZ
    if mod(i,3) ~= 0
        if (Tp(1,i) == 1)
            if (USE_QUANT == 0)
                X_f(ctr_a,:,1:3)=Xin(1:3,i:i+WINDOW_SZ)';
            else
                X_f(ctr_a,:,1:3)=T(1:3,i:i+WINDOW_SZ)';
            end
            Y=[Y;[1 0]];
            ctr_a=ctr_a+1;
            
            if (AUGMENT_DS == 1)
                ii=1;
                while (ii <= AUG_PER_WINDOW)
                    if (USE_QUANT == 0)
                        Mut_X1=gen_variations(Xin(1,i:i+WINDOW_SZ),AFT_EVERY,NCPY_EXT);
                        Mut_X2=gen_variations(Xin(2,i:i+WINDOW_SZ),AFT_EVERY,NCPY_EXT);
                        Mut_X3=gen_variations(Xin(3,i:i+WINDOW_SZ),AFT_EVERY,NCPY_EXT);
                    else
                        Mut_X1=gen_variations(T(1,i:i+WINDOW_SZ),AFT_EVERY,NCPY_EXT);
                        Mut_X2=gen_variations(T(2,i:i+WINDOW_SZ),AFT_EVERY,NCPY_EXT);
                        Mut_X3=gen_variations(T(3,i:i+WINDOW_SZ),AFT_EVERY,NCPY_EXT);
                    end
                    Mut_X=[Mut_X1' Mut_X2' Mut_X3'];
                    X_f(ctr_a,:,1:3)=Mut_X;
                    Y=[Y;[1 0]];
                    ii=ii+1;
                    ctr_a=ctr_a+1;
                end
            end
        else
            if (USE_QUANT == 0)
                X_f(ctr_a,:,1:3)=Xin(1:3,i:i+WINDOW_SZ)';
            else
                X_f(ctr_a,:,1:3)=T(1:3,i:i+WINDOW_SZ)';
            end
            Y=[Y;[0 1]];
            ctr_a=ctr_a+1;
        end
    else
        if (Tp(1,i) == 1)
            if (USE_QUANT == 0)
                XVal_f(ctr_b,:,1:3)=Xin(1:3,i:i+WINDOW_SZ)';
            else
                XVal_f(ctr_b,:,1:3)=T(1:3,i:i+WINDOW_SZ)';
            end
            YVal=[YVal;[1 0]];        
            ctr_b=ctr_b+1;
            
            if (AUGMENT_DS == 1)
                ii=1;
                while (ii <= AUG_PER_WINDOW)
                    if (USE_QUANT == 0)
                        Mut_X1=gen_variations(Xin(1,i:i+WINDOW_SZ),AFT_EVERY,NCPY_EXT);
                        Mut_X2=gen_variations(Xin(2,i:i+WINDOW_SZ),AFT_EVERY,NCPY_EXT);
                        Mut_X3=gen_variations(Xin(3,i:i+WINDOW_SZ),AFT_EVERY,NCPY_EXT);
                    else
                        Mut_X1=gen_variations(T(1,i:i+WINDOW_SZ),AFT_EVERY,NCPY_EXT);
                        Mut_X2=gen_variations(T(2,i:i+WINDOW_SZ),AFT_EVERY,NCPY_EXT);
                        Mut_X3=gen_variations(T(3,i:i+WINDOW_SZ),AFT_EVERY,NCPY_EXT);
                    end
                    Mut_X=[Mut_X1' Mut_X2' Mut_X3'];
                    XVal_f(ctr_b,:,1:3)=Mut_X;
                    YVal=[YVal;[1 0]];
                    ii=ii+1;
                    ctr_b=ctr_b+1;
                end
            end
        else
            if (USE_QUANT == 0)
                XVal_f(ctr_b,:,1:3)=Xin(1:3,i:i+WINDOW_SZ)';
            else
                XVal_f(ctr_b,:,1:3)=T(1:3,i:i+WINDOW_SZ)';
            end
            YVal=[YVal;[0 1]];
            ctr_b=ctr_b+1;
        end

    end  
end

for idx=1:size(Y,1)
    X(idx,:,1:3)=X_f(idx,:,1:3);
end

%Make the validation set have equal number of classes 
tot=0;
last_idx=0;
for idx=1:size(YVal,1)
    if (YVal(idx,1) == 1)
        tot=tot+1;
    end
    last_idx=last_idx+1;
    XVal(idx,:,1:3)=XVal_f(idx,:,1:3);
    if (tot == sum(YVal(:,2)))
        break;
    end
end
YVal=YVal(1:last_idx,:);

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
opt.rmspropDecay = 0.99;%0.75;
% opt.initialMomentum = 0.5; opt.switchEpochMomentum = 1; opt.finalMomentum = 0.9;
opt.learningRate = 0.015;
opt.learningDecaySchedule = 'exp'; %'no_decay';%'t/T';%'step';
opt.learningDecayRate = 1;
%opt.learningDecayRateStep = 5;

opt.dropout = 1;
opt.weightDecay = false;
opt.maxNorm = false;

opt.maxEpochs = 1000;
opt.earlyStoppingPatience = 1000;
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
%[yplabel, confidence, classes, classConfidences, yp] = predict(XVal, @two_ctrnn, model, opt);
[yplabel, confidence, classes, classConfidences, yp] = predict(HXVal, @two_ctrnn, model, opt);

%%
if (SHOW_DEC_BOUND_CLASSIFIER == 1)
    figure;
    mark_sz=1;
    hold on;
    Hctr_b=1;
    for i=1:size(HXVal,1)                  
            if (yplabel(i) == 2)
                plot3(HXVal(Hctr_b,:,1),HXVal(Hctr_b,:,2),HXVal(Hctr_b,:,3),'ro','MarkerSize',mark_sz+8)
            else
                plot3(HXVal(Hctr_b,:,1),HXVal(Hctr_b,:,2),HXVal(Hctr_b,:,3),'b.','MarkerSize',mark_sz)
            end
            
            Hctr_b=Hctr_b+1;            
    end
    hold off;    
end
