[ errTr, errTst ] = linridgereg();

lambdaList = [ -5 -4 -3 -2 -1 0 1 2 3 4];

[ row, col ] = size( errTr );

subplot(1,2,1)
plot(  lambdaList, errTst )
title('Graph of Lambda vs MSE(Test data) for different training set fractions');
xlabel('lambda on the scale of log(base 10)');
ylabel('Mean Squared Error');
hleg = legend('0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9');
set(hleg,'Location','NorthWest');
ylim( [0, 50] );
subplot(1,2,2);
plot( lambdaList, errTr )
title('Graph of Lambda vs MSE(Training data) for different training set fractions');
xlabel('lambda on the scale of log(base 10)');
ylabel('Mean Squared Error');
hleg = legend('0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9');
set(hleg,'Location','NorthWest');
ylim( [0, 50] )
