function [ weights ] = mylinridgereg( X, T, lambda )
    [ row, col ] = size(X);
    weights = ( transpose(X)*X + lambda*eye( col ) ) \ ...
        ( transpose( X ) * T );