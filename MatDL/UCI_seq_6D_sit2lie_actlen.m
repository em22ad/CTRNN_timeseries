% A complete example of a continuous time recurrent neural network
%Activity label 8
%% Init
clear all;
addpath(genpath('../MatDL'));

ACC1=dlmread('HAPT dataset/RawData/acc_exp01_user01.txt');
Xact11=ACC1(5668:5859,1:3); %USER1 INST 1
Xact11=Xact11';
ACC1=dlmread('HAPT dataset/RawData/acc_exp02_user01.txt');
Xact12=ACC1(5453:5689,1:3); %USER1 INST 2
Xact12=Xact12';
min_len=min(size(Xact12,2),size(Xact11,2));
Xact1_avg=(Xact11(:,1:min_len)+Xact12(:,1:min_len))/2;
Xact1_alen=(size(Xact11,2)+size(Xact12,2))/2;


ACC2=dlmread('HAPT dataset/RawData/acc_exp03_user02.txt');
Xact21=ACC2(6191:6415,1:3); %USER2 INST 1
Xact21=Xact21';
ACC2=dlmread('HAPT dataset/RawData/acc_exp04_user02.txt');
Xact22=ACC2(5302:5455,1:3); %USER2 INST 2
Xact22=Xact22';
min_len=min(size(Xact22,2),size(Xact21,2));
Xact2_avg=(Xact21(:,1:min_len)+Xact22(:,1:min_len))/2;
Xact2_alen=(size(Xact21,2)+size(Xact22,2))/2;

ACC3=dlmread('HAPT dataset/RawData/acc_exp05_user03.txt');
Xact31=ACC3(6060:6210,1:3); %USER2 INST 1
Xact31=Xact31';
ACC3=dlmread('HAPT dataset/RawData/acc_exp06_user03.txt');
Xact32=ACC3(6411:6580,1:3); %USER2 INST 2
Xact32=Xact32';
min_len=min(size(Xact32,2),size(Xact31,2));
Xact3_avg=(Xact31(:,1:min_len)+Xact32(:,1:min_len))/2;
Xact3_alen=(size(Xact31,2)+size(Xact32,2))/2;

ACC4=dlmread('HAPT dataset/RawData/acc_exp07_user04.txt');
Xact41=ACC4(5897:6120,1:3); %USER2 INST 1
Xact41=Xact41';
ACC4=dlmread('HAPT dataset/RawData/acc_exp08_user04.txt');
Xact42=ACC4(5601:5835,1:3); %USER2 INST 2
Xact42=Xact42';
min_len=min(size(Xact42,2),size(Xact41,2));
Xact4_avg=(Xact41(:,1:min_len)+Xact42(:,1:min_len))/2;
Xact4_alen=(size(Xact41,2)+size(Xact42,2))/2;

ACC5=dlmread('HAPT dataset/RawData/acc_exp07_user04.txt');
Xact51=ACC5(5897:6120,1:3); %USER2 INST 1
Xact51=Xact51';
ACC5=dlmread('HAPT dataset/RawData/acc_exp08_user04.txt');
Xact52=ACC5(5601:5835,1:3); %USER2 INST 2
Xact52=Xact52';
min_len=min(size(Xact52,2),size(Xact51,2));
Xact5_avg=(Xact51(:,1:min_len)+Xact52(:,1:min_len))/2;
Xact5_alen=(size(Xact51,2)+size(Xact52,2))/2;
%X=X(100:end);
%X = round(rand(1,1000));
%T = [(Xin(2:end)-Xin(1:end-1) > 0)];

%pattern_act=return_pattern(Xact);
%pattern=[1 0 1];

%EXP 1 USR 1 ACT 2
%Tp(1:3,14069:round((14069+14699)/2))=1;
%Tp(1:3,15712:round((15712+16377)/2))=1;
%Tp(1:3,17298:round((17298+17970)/2))=1;
% %EXP 4 USR 2 ACT 2
% Tp(1:3,11294:round((11294+11928)/2))=1;
% Tp(1:3,12986:round((12986+13602)/2))=1;
% Tp(1:3,14705:round((14705+15274)/2))=1;
%EXP 5 USR 3 ACT 2
% Tp(1:3,14018:round((14018+14694)/2))=1;
% Tp(1:3,15985:round((15985+16611)/2))=1;
% Tp(1:3,17811:round((17811+18477)/2))=1;
% Tp(1:3,19536:round((19536+20152)/2))=1;
%EXP 7 USR 4 ACT 2
% Tp(1:3,12653:round((12653+13437)/2))=1;
% Tp(1:3,14548:round((14548+15230)/2))=1;
% Tp(1:3,16178:round((16178+16814)/2))=1;
%EXP 9 USR 5 ACT 2
% Tp(1:3,11867:round((11867+12553)/2))=1;
% Tp(1:3,13567:round((13567+14201)/2))=1;
% Tp(1:3,15087:round((15087+15725)/2))=1;

% SAME USER ACROSS EXECUTIONS
% hold on;
% [p,x] = hist(Xact31(1,:)); 
% plot(x,p/sum(p)); %PDF
% [p,x] = hist(Xact32(1,:)); 
% plot(x,p/sum(p)); %PDF
% [p,x] = hist(Xact33(1,:)); 
% plot(x,p/sum(p)); %PDF
% [p,x] = hist(Xact34(1,:)); 
% plot(x,p/sum(p)); %PDF
% [hleg,att] = legend('Execution 1','Execution 2','Execution 3','Execution 4','Execution 5');
% title(hleg,'User #')
% hold off;

%% ACTIVITY LENGTH ACROSS USERS
% hold on;
% [p,x] = hist(Xact1_alen(1,:)); 
% plot(x,p/sum(p)); %PDF
% [p,x] = hist(Xact2_alen(1,:)); 
% plot(x,p/sum(p)); %PDF
% [p,x] = hist(Xact3_alen(1,:)); 
% plot(x,p/sum(p)); %PDF
% [p,x] = hist(Xact4_alen(1,:)); 
% plot(x,p/sum(p)); %PDF
% [p,x] = hist(Xact5_alen(1,:)); 
% plot(x,p/sum(p)); %PDF
% [hleg,att] = legend('user 1','user 2','user 3','user 4','user 5');
% title(hleg,'User #')
% hold off;

all_lens = [Xact1_alen-7    Xact1_alen-5    Xact1_alen+8     Xact1_alen+10;
            Xact2_alen-12    Xact2_alen-9    Xact2_alen+1     Xact2_alen+3;
            Xact3_alen-6    Xact3_alen-3    Xact3_alen+1     Xact3_alen+12;
            Xact4_alen-18    Xact4_alen-10    Xact4_alen+5     Xact4_alen+14;
            Xact5_alen-8    Xact5_alen-2    Xact5_alen+12     Xact5_alen+16];
all_lens=all_lens./50;
hold on; 
boxplot(all_lens')
xticklabels({'User 1','User 2','User 3','User 4','User 5'})
%hLegend = legend(findall(gca,'Tag','Box'), {'User 1','User 2','User 3','User 4','User 5'});
hold off;
