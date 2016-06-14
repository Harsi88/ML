function hw3part4
close all;
clear all;
clc;
load('mnist.mat');

tridx = [];
validx = [];
tsidx = [];

% fraction of the data to be used for validation and test.
valfrac = 0.2;
tsfrac = 0.2;

for k = 1:size(label,2)
    % ifnd the indices of the data points of a particular class
    r = find(label(:,k) == 1);
    % number of data points belonging tot eh k^th class
    nclass = length(r);
    % randomize the indices for the k^{th} class data points
    ridx = randperm(nclass);
    % use the first nclass*tsfrac indices as the test data points
    temptsidx = r(ridx(1:nclass*tsfrac));
    % use the next nclass*valfrac indices as the validation set
    tempvalidx = r(ridx(nclass*tsfrac+1:nclass*tsfrac+1+nclass*valfrac));
    % use the remaining indices as training points
    temptridx = setdiff(r, [temptsidx; tempvalidx]);    
    % append the indices to the cumulative variable
    tridx = [tridx temptridx];
    tsidx = [tsidx temptsidx];
    validx = [validx tempvalidx];
end

% separate the train, validation and test datasets
trX = data(tridx,:);
trY = label(tridx,:);

valX = data(validx,:);
valY = label(validx,:);

tsX = data(tsidx,:);
tsY = label(tsidx,:);


% size of the training data
[N, D] = size(trX);
% number of epochs
nEpochs = 100;
% learning rate
eta = 0.01;
% number of hidden layer units
H = 500;
% number of output layer units
K = 10;

% randomize the weights from input to hidden layer units
% 'TO DO'
w = -0.3+(0.6)*rand(H,(D+1));
% randomize the weights from hidden to output layer units
% 'TO DO'
v = -0.3+(0.6)*rand(K,(H+1));

% let us create the indices for the batches as it cleans up script later
% size of the training batches
batchsize = 25;
% number of batches
nBatches = floor(N/batchsize);
% create the indices of the data points used for each batch
% i^th row in batchindices will give th eindices for the data points for
% the i^th batch
batchindices = reshape(1:batchsize*nBatches',batchsize, nBatches);
batchindices = batchindices';
% if there are any data points left out, add them at the end padding with
% some other indices from the previous batch
if N - batchsize*nBatches >0
    batchindices(end+1,:)=batchindices(end,:);
    batchindices(end,1:(N - batchsize*nBatches)) = [batchsize*nBatches+1: N];
end

% randomize the order of the training data
ridx = randperm(N);
trX = trX(ridx,:);
trY= trY(ridx,:);

classErrorTr = zeros(nEpochs);
classErrorVal = zeros(nEpochs);
for epoch = 1:nEpochs
    for batch = 1:nBatches
        % Call the forward pass function to obtain the outputs
        % 'TO DO'
        [z, ydash] = forwardpass( trX(batchindices(batch, :), :), w, v );
        % Call the gradient function to obtain the required gradient
        % updates
        % 'TO DO'
        [deltaw, deltav] = computegradient( trX(batchindices(batch,:),:)...
            , trY(batchindices(batch, :), :), w, v, z, ydash);
        % update the weights of the two sets of weights
        % 'TO DO' w = w + eta*deltaw and v = v + eta*deltav
        w = w + eta*deltaw;
        v = v + eta*deltav;
        % at the end of epoch compute the classification error on training
        % and validation dataset
        % 'TO DO'
    end
    [~, ydash] = forwardpass( trX, w, v );
    classErrorTr(epoch) = classerror( trY, ydash );
    [~, ydash] = forwardpass( valX, w, v );
    classErrorVal(epoch) = classerror( valY, ydash );
    fprintf('Classification error after epoch %d: %f, %f\n',epoch,...
        classErrorTr(epoch), classErrorVal(epoch));
end

% compute the classification error on the test set
% 'TO DO'
[~, ydash] = forwardpass( tsX, w, v );
classErrorTs = classerror( tsY, ydash );

plot(1:nEpochs, classErrorTr, 'LineWidth', 1)
xlabel('Number of Epochs')
ylabel('Training Error')
figure;
plot(1:nEpochs, classErrorVal, 'LineWidth', 1)
xlabel('Number of Epochs')
ylabel('Validation Error')

% plot atmost 2 misclassified examples for each digit using the displayData
% function
figure;
final_mat=[];
for k=1:K
    class = find(tsY(:,k)==1);
    misclass = abs(ydash(class,k)-tsY(class,k))>0.00001;
    misclassind = class(find(misclass==1));
    misclassind = misclassind(1:2);
    final_mat = [ final_mat; tsX(misclassind,:) ];
end

displayData(final_mat);

function [z, ydash] = forwardpass(X, w, v)
% this function performs the forward pass on a set of data points in the
% variable X and returns the output of the hidden layer units- z and the
% output layer units ydash
% 'TO DO'

X = [ X, ones(size(X,1),1)];
z = [tanh(X*w'), ones(size(X,1),1)];
ydash = softmax((z*v')')';

return;

function [deltaw, deltav] = computegradient(X, Y, w, v, z, ydash)
% this function computes the gradient of the error function with resepct to
% the weights
% 'TO DO'
X = [ X, ones(size(X,1),1)];
deltav = ((Y-ydash).*ydash.*(1-ydash))'*z;
deltaw = ((Y-ydash)*v(:,1:(size(v,2)-1)).*...
    (1-z(:,1:(size(v,2)-1)).*z(:,1:(size(v,2)-1))))'*X;

return;

function error = classerror(y, ydash)
% this function computes the classification error given the actual output y
% and the predicted output ydash
error = sum(sum(abs(y-ydash), 2)>0.01);
return;

function [h, display_array] = displayData(X)
% DO NOT CHANGE ANYTHING HERE
%DISPLAYDATA Display 2D data in a nice grid
%   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
%   stored in X in a nice grid. It returns the figure handle h and the 
%   displayed array if requested.
%   example to plot the data provided by Andrew Ng.

% Set example_width automatically if not passed in
if ~exist('example_width', 'var') || isempty(example_width) 
	example_width = round(sqrt(size(X, 2)));
end

% Gray Image
colormap(gray);

% Compute rows, cols
[m, n] = size(X);
example_height = (n / example_width);

% Compute number of items to display
display_rows = floor(sqrt(m));
display_cols = ceil(m / display_rows);

% Between images padding
pad = 1;

% Setup blank display
display_array = - ones(pad + display_rows * (example_height + pad), ...
                       pad + display_cols * (example_width + pad));

% Copy each example into a patch on the display array
curr_ex = 1;
for j = 1:display_rows
	for i = 1:display_cols
		if curr_ex > m, 
			break; 
		end
		% Copy the patch
		
		% Get the max value of the patch
		max_val = max(abs(X(curr_ex, :)));
		display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
		              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
						reshape(X(curr_ex, :), example_height, example_width) / max_val;
		curr_ex = curr_ex + 1;
	end
	if curr_ex > m, 
		break; 
	end
end

% Display Image
h = imagesc(display_array, [-1 1]);

% Do not show axis
axis image off

drawnow;
return;
