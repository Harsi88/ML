function [] = plotdecisionboundary(w, X, Y, color)
    x1 = X';
    x2 = Y';
    
    sz = size(x1,1);
    y = zeros(sz,sz);
    
    for i =1:sz
        X = [x1(i,1)*ones(size(x1,1),1) x2];
        newX = featuretransform(X,2);
        y(:,i) = (1+exp(-newX*w')).^-1;
    end
    
    contour(x1,x2,y,0.5, color)
    title('Contour for Logistic Regression')
