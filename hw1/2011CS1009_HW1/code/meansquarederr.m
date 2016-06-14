function [ err ] = meansquarederr( T, Tdash )
    [ row, col ] = size( T );
    
    err = 0;
    for idx = 1:row
        err = err + ( T( idx ) - Tdash( idx ) )^2;
    end
    
    err = err/(2*row);