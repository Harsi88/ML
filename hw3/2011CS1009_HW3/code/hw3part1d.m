% This function is the primary driver for homework 3 part 1
function hw3part1d
close all;
clear all;
clc;
% we will experiment with a simple 2d dataset to visualize the decision
% boundaries learned by a MLP. Our goal is to study the changes to the
% decision boundary and the training error with respect to the following
% parameters
% - increasing overlap between the data points of the different classes
% - increasing the number of training iterations
% - increase the number of hidden layer neurons
% - see the effect of learning rate on the convergence of the network


% centroid for the three classes
c1=[1 1];
c2=[3 1];
c3=[2 3];

% standard deviation for the three classes
% "increase this quantity to increase the overlap between the classes"
% change this quantity to 0.75 when solving 1(f).
sd=0.2;

% number of data points per class
N=100;

rand('seed', 1);

% generate data points for the three classes
x1=randn(N,2)*sd+ones(N,1)*c1;
x2=randn(N,2)*sd+ones(N,1)*c2;
x3=randn(N,2)*sd+ones(N,1)*c3;

% generate the labels for the three classes in the binary notation
y1= repmat([1 0 0],N,1);
y2= repmat([0 1 0],N,1);
y3= repmat([0 0 1],N,1);

% creating the test data points
a1min = min([x1(:,1);x2(:,1);x3(:,1)]);
a1max = max([x1(:,1);x2(:,1);x3(:,1)]);

a2min = min([x1(:,2);x2(:,2);x3(:,2)]);
a2max = max([x1(:,2);x2(:,2);x3(:,2)]);

[a1, a2] = meshgrid(a1min:0.1:a1max, a2min:0.1:a2max);

testX=[a1(:) a2(:)];

% Experimenting with MLP

% number of epochs for training
nEpochs = 1000;

% learning rate
eta = 0.01;

figure;
hold all
i = 1;
h = zeros(nEpochs,3);
% H is number of hidden layers
for H = [2 4 8 16 32 64 128]
    % train the MLP using the generated sample dataset
    [w, v, trainerror] = mlptrain([x1;x2;x3],[y1;y2;y3], H, eta, nEpochs);
    if H==2
        w2=w;
        v2=v;
    elseif H==4
        w4=w;
        v4=v;
    elseif H==64
        w64=w;
        v64=v;
    end
    % plot the train error againt the number of epochs
    plot(1:nEpochs, trainerror, ':', 'LineWidth', 2)
    h(:,i) = plot(1:nEpochs, trainerror, ':', 'LineWidth', 2);
    i=i+1;
end
xlabel('Number of Epochs')
ylabel('Mean Training Error')
legend(h(1,:),{'H=2','H=4','H=8','H=16','H=32','H=64','H=128'})
hold off

ydash_2 = mlptest(testX, w2, v2);
ydash_4 = mlptest(testX, w4, v4);
ydash_64 = mlptest(testX, w64, v64);

[~, idx2] = max(ydash_2, [], 2);
[~, idx4] = max(ydash_4, [], 2);
[~, idx64] = max(ydash_64, [], 2);

label2 = reshape(idx2, size(a1));
label4 = reshape(idx4, size(a1));
label64 = reshape(idx64, size(a1));

% ploting the approximate decision boundary
% -------------------------------------------

figure;
imagesc([a1min a1max], [a2min a2max], label2), hold on,
set(gca, 'ydir', 'normal'),
% colormap for the classes:
% class 1 = light red, 2 = light green, 3 = light blue
cmap = [1 0.8 0.8; 0.9 1 0.9; 0.9 0.9 1];
colormap(cmap);
% plot the training data
plot(x1(:,1),x1(:,2),'r.', 'LineWidth', 2),
plot(x2(:,1),x2(:,2),'g+', 'LineWidth', 2),
plot(x3(:,1),x3(:,2),'bo', 'LineWidth', 2),
legend('Class 1', 'Class 2', 'Class 3', 'Location', 'NorthOutside', ...
    'Orientation', 'horizontal');
hold off
%---------------------------------------------
figure;
imagesc([a1min a1max], [a2min a2max], label4), hold on,
set(gca, 'ydir', 'normal'),
% colormap for the classes:
% class 1 = light red, 2 = light green, 3 = light blue
cmap = [1 0.8 0.8; 0.9 1 0.9; 0.9 0.9 1];
colormap(cmap);
% plot the training data
plot(x1(:,1),x1(:,2),'r.', 'LineWidth', 2),
plot(x2(:,1),x2(:,2),'g+', 'LineWidth', 2),
plot(x3(:,1),x3(:,2),'bo', 'LineWidth', 2),
legend('Class 1', 'Class 2', 'Class 3', 'Location', 'NorthOutside', ...
    'Orientation', 'horizontal');
hold off
%------------------------------------------
figure;
imagesc([a1min a1max], [a2min a2max], label64), hold on,
set(gca, 'ydir', 'normal'),
% colormap for the classes:
% class 1 = light red, 2 = light green, 3 = light blue
cmap = [1 0.8 0.8; 0.9 1 0.9; 0.9 0.9 1];
colormap(cmap);
% plot the training data
plot(x1(:,1),x1(:,2),'r.', 'LineWidth', 2),
plot(x2(:,1),x2(:,2),'g+', 'LineWidth', 2),
plot(x3(:,1),x3(:,2),'bo', 'LineWidth', 2),
legend('Class 1', 'Class 2', 'Class 3', 'Location', 'NorthOutside', ...
    'Orientation', 'horizontal');
hold off

% viewing the decision surface for the three classes
% ydash1 = reshape(ydash_2(:,1), size(a1));
% ydash2 = reshape(ydash_2(:,2), size(a1));
% ydash3 = reshape(ydash_2(:,3), size(a1));
% 
% figure;
% surf(a1, a2, ydash1, 'FaceColor', [1 0 0], 'FaceAlpha', 0.5), hold on,...
% surf(a1, a2, ydash2, 'FaceColor', [0 1 0], 'FaceAlpha', 0.5), hold on,...
% surf(a1, a2, ydash3, 'FaceColor', [0 0 1], 'FaceAlpha', 0.5);

function [w, v, trainerror] = mlptrain(X, Y, H, eta, nEpochs)
% X - training data of size NxD
% Y - training labels of size NxK
% H - the number of hiffe
% eta - the learning rate
% nEpochs - the number of training epochs
% define and initialize the neural network parameters

trainerror = zeros(nEpochs);

% number of training data points
N = size(X,1);
% number of inputs
D = size(X,2); % excluding the bias term
% number of outputs
K = size(Y,2);

% weights for the connections between input and hidden layer
% random values from the interval [-0.3 0.3]
% w is a Hx(D+1) matrix
w = -0.3+(0.6)*rand(H,(D+1));

% weights for the connections between input and hidden layer
% random values from the interval [-0.3 0.3]
% v is a Kx(H+1) matrix
v = -0.3+(0.6)*rand(K,(H+1));

% randomize the order in which the input data points are presented to the
% MLP
iporder = randperm(N);

% mlp training through stochastic gradient descent
for epoch = 1:nEpochs
    z = ones(N, H+1);
    ydash = ones(N, K);
    for n = 1:N
        % the current training point is X(iporder(n), :)
        % forward pass
        % --------------
        % input to hidden layer
        % calculate the output of the hidden layer units - z
        % ---------
        %'TO DO'%
        padded_X = [X(iporder(n), :), 1];
        z(n,:) = [logsig(padded_X*w'), 1];
        % ---------
        % hidden to output layer
        % calculate the output of the output layer units - ydash
        % ---------
        %'TO DO'% soft max la
        ydash(n,:) = softmax((z(n,:)*v')')';
        % ---------
        % backward pass
        % ---------------
        % update the weights for the connections between hidden and
        % outlayer units
        % ---------
        %'TO DO'%
        deltaV = eta.*( ((Y(iporder(n),:)-ydash(n,:)).*(1-ydash(n,:)).*...
            ydash(n,:))'*z(n,:) );
        % ---------
        % update the weights for the connections between the input and
        % hidden later units
        % ---------
        %'TO DO'%
        deltaW = eta.*(((Y(iporder(n),:)-ydash(n,:)).*(1-ydash(n,:))...
            .*ydash(n,:)*v(:,1:(size(v,2)-1))).*z(n,1:(size(z,2)-1)).*...
            (1-z(n,1:(size(z,2)-1))))'*padded_X;
        % ---------
        
        v = v + deltaV;
        w = w + deltaW;
        
    end
    ydash = mlptest(X, w, v);
    % compute the training error
    % ---------
    %'TO DO'% uncomment the next line after adding the necessary code
    trainerror(epoch) = mean(-sum(Y.*log(ydash),2));
    % ---------
    fprintf('training error after epoch %d: %f\n',epoch,...
        trainerror(epoch));
end
return;

function ydash = mlptest(X, w, v)
% forward pass of the network

% forward pass to estimate the outputs
% --------------------------------------
% input to hidden for all the data points
% calculate the output of the hidden layer units
% ---------
%'TO DO'%
padded_X = [ X, ones(size(X,1),1)];
z = [logsig(padded_X*w'), ones(size(X,1),1)];
% ---------% hidden to output for all the data points
% calculate the output of the output layer units
% ---------
%'TO DO'%
ydash = softmax((z*v')')';
% ---------
return;
