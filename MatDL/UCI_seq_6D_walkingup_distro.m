% A complete example of a continuous time recurrent neural network
%Activity label 2
%% Init
clear all;
addpath(genpath('../MatDL'));
%% Generate NARMA Series
% N=300;
% Xin = gen_narma_30(N);
% Xin=Xin(100:end);
% %X = round(rand(1,1000));
% T = [(Xin(2:end)-Xin(1:end-1) > 0)];

% ACC1=dlmread('HAPT dataset/RawData/gyro_exp01_user01.txt');
% ACC2=dlmread('HAPT dataset/RawData/gyro_exp04_user02.txt');
% ACC3=dlmread('HAPT dataset/RawData/gyro_exp05_user03.txt');
% ACC4=dlmread('HAPT dataset/RawData/gyro_exp07_user04.txt');
% ACC5=dlmread('HAPT dataset/RawData/gyro_exp09_user05.txt');
ACC1=dlmread('HAPT dataset/RawData/gyro_exp01_user01.txt');
ACC2=dlmread('HAPT dataset/RawData/gyro_exp04_user02.txt');
ACC3=dlmread('HAPT dataset/RawData/gyro_exp05_user03.txt');
ACC4=dlmread('HAPT dataset/RawData/gyro_exp07_user04.txt');
ACC5=dlmread('HAPT dataset/RawData/gyro_exp09_user05.txt');

Xact13=ACC1(14069:14699,1:3); %USER1 INST 3
Xact13=Xact13';
Xact12=ACC1(15712:16377,1:3); %USER1 INST 2
Xact12=Xact12';
Xact11=ACC1(17298:17970,1:3); %USER1 INST 1
Xact11=Xact11';
min_len=min(min(size(Xact12,2),size(Xact13,2)),size(Xact11,2));
Xact1_avg=(Xact11(:,1:min_len)+Xact12(:,1:min_len)+Xact13(:,1:min_len))/3;


Xact23=ACC2(14705:15274,1:3); %USER2 INST 3
Xact23=Xact23';
Xact22=ACC2(12986:13602,1:3); %USER2 INST 2
Xact22=Xact22';
Xact21=ACC2(11294:11928,1:3); %USER2 INST 1
Xact21=Xact21';
min_len=min(min(size(Xact22,2),size(Xact23,2)),size(Xact21,2));
Xact2_avg=(Xact21(:,1:min_len)+Xact22(:,1:min_len)+Xact23(:,1:min_len))/3;

Xact34=ACC3(19536:20152,1:3); %USER3 INST 3
Xact34=Xact34';
Xact33=ACC3(14018:14694,1:3); %USER3 INST 3
Xact33=Xact33';
Xact32=ACC3(15985:16611,1:3); %USER3 INST 2
Xact32=Xact32';
Xact31=ACC3(17811:18477,1:3); %USER3 INST 1
Xact31=Xact31';
min_len=min(min(min(size(Xact32,2),size(Xact33,2)),size(Xact31,2)),size(Xact34,2));
Xact3_avg=(Xact31(:,1:min_len)+Xact32(:,1:min_len)+Xact33(:,1:min_len)+Xact34(:,1:min_len))/4;

Xact43=ACC4(12653:13437,1:3); %USER2 INST 3
Xact43=Xact43';
Xact42=ACC4(14548:15230,1:3); %USER2 INST 2
Xact42=Xact42';
Xact41=ACC4(16178:16814,1:3); %USER2 INST 1
Xact41=Xact41';
min_len=min(min(size(Xact42,2),size(Xact43,2)),size(Xact41,2));
Xact4_avg=(Xact41(:,1:min_len)+Xact42(:,1:min_len)+Xact43(:,1:min_len))/3;

Xact53=ACC5(11867:12553,1:3); %USER2 INST 3
Xact53=Xact53';
Xact52=ACC5(13567:14201,1:3); %USER2 INST 2
Xact52=Xact52';
Xact51=ACC5(15087:15725,1:3); %USER2 INST 1
Xact51=Xact51';
min_len=min(min(size(Xact52,2),size(Xact53,2)),size(Xact51,2));
Xact5_avg=(Xact51(:,1:min_len)+Xact52(:,1:min_len)+Xact53(:,1:min_len))/3;
%Xact=normalize(Xact);
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
hold on;
[p,x] = hist(Xact31(1,:)); 
plot(x,p/sum(p)); %PDF
[p,x] = hist(Xact32(1,:)); 
plot(x,p/sum(p)); %PDF
[p,x] = hist(Xact33(1,:)); 
plot(x,p/sum(p)); %PDF
[p,x] = hist(Xact34(1,:)); 
plot(x,p/sum(p)); %PDF
[hleg,att] = legend('Execution 1','Execution 2','Execution 3','Execution 4','Execution 5');
title(hleg,'User #')
hold off;

%% ACROSS USERS
% hold on;
% [p,x] = hist(Xact1_avg(1,:)); 
% plot(x,p/sum(p)); %PDF
% [p,x] = hist(Xact2_avg(1,:)); 
% plot(x,p/sum(p)); %PDF
% [p,x] = hist(Xact3_avg(1,:)); 
% plot(x,p/sum(p)); %PDF
% [p,x] = hist(Xact4_avg(1,:)); 
% plot(x,p/sum(p)); %PDF
% [p,x] = hist(Xact5_avg(1,:)); 
% plot(x,p/sum(p)); %PDF
% [hleg,att] = legend('user 1','user 2','user 3','user 4','user 5');
% title(hleg,'User #')
% hold off;

% [f,x] = ecdf(Xact(1,:)); 
% plot(x,f); %CDF