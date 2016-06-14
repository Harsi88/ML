close all; clear all;

rand('seed',1);

% ---- Part a ----
% PCA - Algorithm
% Create N X D data matrix X, with one row vector xn per data point
N = 60;
D = 50;
% classOne = normrnd(2,0.5,[N/3 D]);
% classTwo = normrnd(4,0.5,[N/3 D]);
% classThree = normrnd(6,0.5,[N/3 D]);

% figure;
% hold on;
% plot(classOne(:,1),classOne(:,2),'r.');
% plot(classTwo(:,1),classTwo(:,2),'g.');
% plot(classThree(:,1),classThree(:,2),'b.');
% hold off;

% X = [classOne;classTwo;classThree];
% Y = X;
% Subtract mean xbar from each row vector xn in X

load('hw6q3.mat');
X = Y;
% ---- Part b ----
Xbar = mean(X,1);
X = X - repmat(Xbar,[N,1]);

% sigma is covariance matrix of X
sigma = cov(X);
% find eigenvectors U and eigenvalues A of sigma
% PC's Uk is the K eigenvectors with largest eigenvalues
[U, A] = eigs(sigma,2);
% Transformed data Z = U(t,k)*X
Z = X*U;

PCA = 6*U*(A.^0.5);

figure;
hold on;
plot(X(1:20,1),X(1:20,2),'r+');
plot(X(21:40,1),X(21:40,2),'g+');
plot(X(41:60,1),X(41:60,2),'b+');
plot([-PCA(1,1),PCA(1,1)],[-PCA(2,1),PCA(2,1)],'r');
plot([-PCA(1,2),PCA(1,2)],[-PCA(2,2),PCA(2,2)],'k');
xlim([-4,4]);
title('Question3 - Part b');
hold off;

% hold on;
% plot(Z(1:20,1),Z(1:20,2),'r+');
% plot(Z(21:40,1),Z(21:40,2),'g+');
% plot(Z(41:60,1),Z(41:60,2),'b+');
% hold off;


% ---- Part c ----
k = 3;
means = rand(k,D);
newmeans = zeros(k,D);

class = zeros(1,N);
prevClass = zeros(1,N);
while(1)
prevClass = class;
for i=1:N
   minidx = 1;
   min = norm(means(1,:)-X(i,:));
   for j=2:k
       if(min>norm(means(j,:)-X(i,:)))
           min = norm(means(j,:)-X(i,:));
           minidx = j;
       end
   end
   class(1,i) = minidx;
end
for i=1:k
    idx = find(class(1,:)==i);
    newmeans(i,:) = mean(X(idx,:),1);
end
if(prevClass==class)
    break;
else
    means = newmeans;
end
end

figure;
hold on;
col = ['r','g','b','k'];
for i=1:k
    idx = find(class(1,:)==i);
    temp = (X(idx,:));
    scatter(temp(:,1),temp(:,2),col(i));
end
title('Question3 - Part c : Examples labelled using 3-means clustering');
hold off;

% ---- Part d ----
k = 2;
means = rand(k,D);
newmeans = zeros(k,D);

while(1)
prevClass = class;
for i=1:N
   minidx = 1;
   min = norm(means(1,:)-X(i,:));
   for j=2:k
       if(min>norm(means(j,:)-X(i,:)))
           min = norm(means(j,:)-X(i,:));
           minidx = j;
       end
   end
   class(1,i) = minidx;
end
for i=1:k
    idx = find(class(1,:)==i);
    newmeans(i,:) = mean(X(idx,:),1);
end
if(prevClass==class)
    break;
else
    means = newmeans;
end
end

figure;
hold on;
col = ['r','g','b','k'];
for i=1:k
    idx = find(class(1,:)==i);
    temp = (X(idx,:));
    scatter(temp(:,1),temp(:,2),col(i));
end
title('Question3 - Part d : Examples labelled using 2-means clustering');
hold off;

% ---- Part e ----
k = 4;
means = rand(k,D);
newmeans = zeros(k,D);

while(1)
prevClass = class;
for i=1:N
   minidx = 1;
   min = norm(means(1,:)-X(i,:));
   for j=2:k
       if(min>norm(means(j,:)-X(i,:)))
           min = norm(means(j,:)-X(i,:));
           minidx = j;
       end
   end
   class(1,i) = minidx;
end
for i=1:k
    idx = find(class(1,:)==i);
    newmeans(i,:) = mean(X(idx,:),1);
end
if(prevClass==class)
    break;
else
    means = newmeans;
end
end

figure;
hold on;
col = ['r','g','b','k'];
for i=1:k
    idx = find(class(1,:)==i);
    temp = (X(idx,:));
    scatter(temp(:,1),temp(:,2),col(i));
end
title('Question3 - Part e : Examples labelled using 4-means clustering');
hold off;

% ---- Part f ----
k = 3;
DD = 2;
means = rand(k,DD);
newmeans = zeros(k,DD);
Xnew = [X(:,1),X(:,2)];

while(1)
prevClass = class;
for i=1:N
   minidx = 1;
   min = norm(means(1,:)-Xnew(i,:));
   for j=2:k
       if(min>norm(means(j,:)-Xnew(i,:)))
           min = norm(means(j,:)-Xnew(i,:));
           minidx = j;
       end
   end
   class(1,i) = minidx;
end
for i=1:k
    idx = find(class(1,:)==i);
    newmeans(i,:) = mean(Xnew(idx,:),1);
end
if(prevClass==class)
    break;
else
    means = newmeans;
end
end

figure;
hold on;
col = ['r','g','b','k'];
for i=1:k
    idx = find(class(1,:)==i);
    temp = (Xnew(idx,:));
    scatter(temp(:,1),temp(:,2),col(i));
end
title('Question3 - Part f : Examples labelled using 3-means clustering on 2-D data');
hold off;

% ---- Part g ----
k = 3;
means = rand(k,D);
newmeans = zeros(k,D);
Xnew = X./repmat(std(X,1),[N,1]);

while(1)
prevClass = class;
for i=1:N
   minidx = 1;
   min = norm(means(1,:)-Xnew(i,:));
   for j=2:k
       if(min>norm(means(j,:)-Xnew(i,:)))
           min = norm(means(j,:)-Xnew(i,:));
           minidx = j;
       end
   end
   class(1,i) = minidx;
end
for i=1:k
    idx = find(class(1,:)==i);
    newmeans(i,:) = mean(Xnew(idx,:),1);
end
if(prevClass==class)
    break;
else
    means = newmeans;
end
end

figure;
hold on;
col = ['r','g','b','k'];
for i=1:k
    idx = find(class(1,:)==i);
    temp = (Xnew(idx,:));
    scatter(temp(:,1),temp(:,2),col(i));
end
title('Question3 - Part g : Examples labelled using 3-means clustering on scaled data');
hold off;
