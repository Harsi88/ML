close all, clear all;

load('abalone.mat');

[ row, col ] = size(abalone);
frac = 0.3;
lambda = 1;
abaloneTraining = [];
abaloneTest = [];

for idx = 1:row
   toss = rand();
   if toss < frac
       abaloneTraining = [ abaloneTraining; abalone( idx, : ) ];
   else
       abaloneTest = [ abaloneTest; abalone( idx, : ) ];
   end
end

tgtValueTraining = abaloneTraining( :, 9);
tgtValueTest = abaloneTest( :, 9);

[ rowTr, colTr ] = size( abaloneTraining );
[ rowTst, colTst ] = size( abaloneTest );

Xtr = zeros( rowTr, 11 );
Xtr( :, 1 ) = 1;
Xtst = zeros( rowTst, 11);
Xtst( :, 1 ) = 1;

% Vectorizing the first attribute into binary form
for row_idx = 1:rowTr
    if abaloneTraining( row_idx, 1 ) == 0
        Xtr( row_idx, 2 ) = 1;
    elseif abaloneTraining( row_idx, 1 ) == 1
        Xtr( row_idx, 3 ) = 1;
    elseif abaloneTraining( row_idx, 1 ) == 2
        Xtr( row_idx, 4 ) = 1;
    end
end

for row_idx = 1:rowTst
    if abaloneTest( row_idx, 1 ) == 0
        Xtst( row_idx, 2 ) = 1;
    elseif abaloneTest( row_idx, 1 ) == 1
        Xtst( row_idx, 3 ) = 1;
    elseif abaloneTest( row_idx, 1 ) == 2
        Xtst( row_idx, 4 ) = 1;
    end
end

% To store mean and standard deviation for each attribute
% which will be used to standardize the test input
col_mean = zeros( 1, 11 );
col_std = zeros( 1, 11 );

% standardizing Training data
for col_idx = 2:8
    col_mean( 1, col_idx + 3 ) = mean( abaloneTraining( :, col_idx ) );
    col_std( 1, col_idx + 3 ) = std( abaloneTraining( :, col_idx ) );
    for row_idx = 1:rowTr
        Xtr( row_idx, col_idx + 3 ) = ( abaloneTraining( row_idx, col_idx ) - ...
            col_mean( 1, col_idx + 3 ) ) / col_std( 1, col_idx + 3 );
    end
end

% standardizing vector form of first attribute on Training data
for col_idx = 2:4
    col_mean( 1, col_idx ) = mean( Xtr( :, col_idx ) );
    col_std( 1, col_idx ) = std( Xtr( :, col_idx ) );
    for row_idx = 1:rowTr
        Xtr( row_idx, col_idx ) = ( Xtr( row_idx, col_idx ) - ...
            col_mean( 1, col_idx ) ) / col_std( 1, col_idx );
    end
end

% standardizing Test data
for col_idx = 2:8
    for row_idx = 1:rowTst
        Xtst( row_idx, col_idx + 3 ) = ( abaloneTest( row_idx, col_idx ) - ...
            col_mean( 1, col_idx + 3 ) ) / col_std( 1, col_idx + 3 );
    end
end

% standardizing vector form of first attribute on Test data
for col_idx = 2:4
    for row_idx = 1:rowTst
        Xtst( row_idx, col_idx ) = ( Xtst( row_idx, col_idx ) - ...
            col_mean( 1, col_idx ) ) / col_std( 1, col_idx );
    end
end

weight = mylinridgereg( Xtr, tgtValueTraining, lambda );
predictedValuesTst = mylinridgeregeval( Xtst, weight );
meansquarederr( tgtValueTest, predictedValuesTst )

weights = weight;
[ row, col ] = size(weights);
for i = 1:col
    weights(1,i) = abs(weights(1,i));
end

rem_idx = zeros( 1, 3 );
maxx = max( weights );
for i = 1:3
    [temp,rem_idx(i)] = min( weights );
    weights(rem_idx(i)) = maxx;
end

rem_idx = sort(rem_idx);
for i = 1:3
    Xtr(:,rem_idx(4-i)) = [];
    Xtst(:,rem_idx(4-i)) = [];
end

weights = mylinridgereg( Xtr, tgtValueTraining, lambda );
predictedValuesTst = mylinridgeregeval( Xtst, weights );
meansquarederr( tgtValueTest, predictedValuesTst )
