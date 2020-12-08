acc10=[84.59 95.70 98.83 98.73 100.0 100.0];
length=[2 3 4 5 6 7];

acc20=[84.77 96.28 99.17 99.35 100.0 100.0];

acc30=[84.23 96.26 98.93 99.41 100.0 100.0];

acc_=[100 100 100 100 100 100];
length_=[2 3 4 5 6 7];

plot(length,acc10,length,acc20,length,acc30,length_,acc_,'LineWidth',1)
xticks(length)
xticklabels({'2','3','4','5','6','7'})
legend('Elman on NARMA-10','Elman on NARMA-20','Elman on NARMA-30','LSTM on all NARMA')
xlabel('Pattern length') 
ylabel('Detection accuracy')
ylim([80 103])

