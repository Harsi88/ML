function [newX] = featuretransform(X, degree)
    szx = size(X, 1);
    newX = ones(szx,1);
    idx = 2;
    for i = 1:degree
       for j = 0:i 
          for x = 1:szx
              newX(x, idx) = X(x, 1)^(i-j) * X(x, 2)^j;
          end
          idx = idx + 1;
       end
    end
