load('credit.mat');

hold on
for i = 1:100
    if label(i) == 1
        plot(data(i,1), data(i,2), 'r.')
    else
        plot(data(i,1), data(i,2), 'b.')
    end
end

X = featuretransform(data, 2);
[sx, sy] = size(X);
weights = zeros(1,sy);
options = optimset('GradObj', 'on', 'MaxIter', 1000 );
[weight1, objval1] = fminunc(@(weights)...
    (objgradcompute(weights,X,label,0)), weights, options);
plotdecisionboundary(weight1, 0:0.1:7, 0:0.1:7, 'r');

[weight2, objval2] = fminunc(@(weights)...
    (objgradcompute(weights,X,label,0.01)), weights, options);
plotdecisionboundary(weight2, 0:0.1:7, 0:0.1:7, 'b');

[weight3, objval3] = fminunc(@(weights)...
    (objgradcompute(weights,X,label,0.1)), weights, options);
plotdecisionboundary(weight3, 0:0.1:7, 0:0.1:7, 'g');

legend('Red: lambda=0','Blue: lambda=0.01','Green: lambda=0.1')

figure

lindiscriminant(data, label, 'k')
