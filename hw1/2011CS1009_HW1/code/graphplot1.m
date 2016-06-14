[ errTr, errTst ] = linridgereg();

lambdaList = [ -5 -4 -3 -2 -1 0 1 2 3 4];
fraction = 0.1:0.1:0.9;

[ row, col ] = size( errTst );

minerrTst = zeros( 1, row );
idx = zeros( 1, row );

for i = 1:row
    [ minerrTst( 1, i ), idx( 1, i ) ] = min( errTst( i, : ) );
end

for i = 1:row
    idx( 1, i ) = idx( 1, i ) - 5;
end

subplot(1,2,1)
plot( fraction , minerrTst )
xlabel('fraction');
ylabel('mean average mean squared testing error')
title('Graph of fraction vs mean average mean squared testing error');

subplot(1,2,2);
bar( fraction, idx, 0.3)
xlabel('fraction');
ylabel('lambda on the scale of log(base 10)');
title('Graph of fraction vs lambda( for which mean squared error is minimum)');
