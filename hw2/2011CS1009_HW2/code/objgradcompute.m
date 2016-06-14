function [objval, gradval] = objgradcompute(w, X, Y, lambda)
    g = (1+exp(-1*X*w')).^-1;
    objval = -1*mean( Y.*log(g) + (1-Y).*log(1-g) + lambda*sum(w.^2)/2);
    gradval = ((g - Y)'*X + lambda*w)./size(X,1);