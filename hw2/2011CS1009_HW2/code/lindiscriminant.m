function [] = lindiscriminant(X, Y, color)

degree = 3;

newX = featuretransform(X,degree);
newX(:,1) = [];

Mean = zeros(2,size(newX,2));
n1=0;
n2=0;
for i = 1:size(newX,1)
    if(Y(i)==0)
        n1 = n1 + 1;
        Mean(1,:) = Mean(1,:) + newX(i,:);
    else
        n2 = n2 + 1;
        Mean(2,:) = Mean(2,:) + newX(i,:);
    end
end

Mean(1,:) = Mean(1,:)./n1;
Mean(2,:) = Mean(2,:)./n2;

sigma = zeros(size(newX,2),size(newX,2));
for i = 1:size(newX,1)
    sigma = sigma + (newX(i,:)-Mean(1,:))'*(newX(i,:)-Mean(1,:)) + ...
        (newX(i,:)-Mean(2,:))'*(newX(i,:)-Mean(2,:));
end

sigma = sigma./(size(X,1)-size(X,2));

x1 = [0:0.1:7]';
x2 = [0:0.1:7]';
sz = size(x1,1);

y1 = zeros(sz,sz);
y2 = zeros(sz,sz);

hold on
for i = 1:size(X,1)
    if Y(i) == 1
        plot(X(i,1), X(i,2), 'r.')
    else
        plot(X(i,1), X(i,2), 'b.')
    end
end

for i =1:sz
   XX = [x1(i,1)*ones(size(x1,1),1) x2];
   newXX = featuretransform(XX,degree);
   newXX(:,1) = [];
   for j = 1:size(newXX,1)
       y1(i,j) = newXX(j,:)*(sigma\Mean(1,:)') - 0.5*Mean(1,:)*...
           (sigma\Mean(1,:)') + log(n1/size(Y,1));
       y2(i,j) = newXX(j,:)*(sigma\Mean(2,:)') - 0.5*Mean(2,:)*...
           (sigma\Mean(2,:)') + log(n2/size(Y,1));
   end
end

y = abs(y1-y2);

contour(x1,x2,y',0.28, color)
title('Contour for Linear Discriminant')