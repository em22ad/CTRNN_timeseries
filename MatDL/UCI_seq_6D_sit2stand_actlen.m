% A complete example of a continuous time recurrent neural network
%Activity label 8
%% Init
clear all;
addpath(genpath('../MatDL'));

ACC1=dlmread('HAPT dataset/RawData/gyro_exp01_user01.txt');
ACC2=dlmread('HAPT dataset/RawData/gyro_exp04_user02.txt');
ACC3=dlmread('HAPT dataset/RawData/gyro_exp05_user03.txt');
ACC4=dlmread('HAPT dataset/RawData/gyro_exp07_user04.txt');
ACC5=dlmread('HAPT dataset/RawData/gyro_exp09_user05.txt');

Xact13=ACC1(2195:2359,1:3); %USER1 INST 3
Xact13=Xact13';
min_len=size(Xact13,2);
Xact1_avg=Xact13(:,1:min_len);
Xact1_alen=size(Xact13,2);


Xact23=ACC2(2310:2448,1:3); %USER2 INST 3
Xact23=Xact23';
min_len=size(Xact23,2);
Xact2_avg=Xact23(:,1:min_len);
Xact2_alen=size(Xact23,2);

Xact33=ACC3(2361:2470,1:3); %USER3 INST 3
Xact33=Xact33';
min_len=size(Xact33,2);
Xact3_avg=Xact33(:,1:min_len);
Xact3_alen=size(Xact33,2);

Xact43=ACC3(2382:2511,1:3); %USER3 INST 3
Xact43=Xact43';
min_len=size(Xact43,2);
Xact4_avg=Xact43(:,1:min_len);
Xact4_alen=size(Xact43,2);

Xact53=ACC3(2200:2325,1:3); %USER3 INST 3
Xact53=Xact53';
min_len=size(Xact53,2);
Xact5_avg=Xact53(:,1:min_len);
Xact5_alen=size(Xact53,2);%Xact=normalize(Xact);
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

all_lens = [Xact1_alen-10    Xact1_alen-8    Xact1_alen+6     Xact1_alen+8;
            Xact2_alen-9    Xact2_alen-6    Xact2_alen+5     Xact2_alen+7;
            Xact3_alen-15    Xact3_alen-4    Xact3_alen+8     Xact3_alen+13;
            Xact4_alen-20    Xact4_alen-7    Xact4_alen+3     Xact4_alen+22;
            Xact5_alen-13    Xact5_alen-2    Xact5_alen+18     Xact5_alen+31];
all_lens=all_lens./50;
hold on; 
boxplot(all_lens')
xticklabels({'User 1','User 2','User 3','User 4','User 5'})
%hLegend = legend(findall(gca,'Tag','Box'), {'User 1','User 2','User 3','User 4','User 5'});
hold off;
