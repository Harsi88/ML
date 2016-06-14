% Import the file
load('-mat', 'dataset1.mat'); % variable names are X and Y

rand('seed',0.001);
% number of total iterations
T = 500;
Nd = size(X, 1);
N = Nd-20; % due to stratification
% Initialise weights
D = ones(N,1)./N;
alpha = zeros(T,1);
hypothesis = zeros(N,T);
testhypothesis = zeros(Nd-N,T);
kfold = 10;
c = cvpartition(Y,'kfold',kfold);
finalhypothesisTr = zeros(N,kfold);
finalhypothesisTst = zeros(Nd-N,kfold);
acc = zeros(kfold,T);
testAcc = zeros(1,kfold);
trainAcc = zeros(1,kfold);
confMat = zeros(3,3,kfold);
avgConfMat = zeros(3,3);
precision = zeros(2,kfold);
avgPrecision = zeros(1,2);
recall = zeros(2,kfold);
avgRecall = zeros(1,2);

for k = 1:kfold
    idxTr = find(c.training(k)==1);
    Xtr = X(idxTr,:);
    Ytr = Y(idxTr);
    idxTest = find(c.test(k)==1);
    Xtst = X(idxTest,:);
    Ytst = Y(idxTest);
    
    for i = 1:T
        % Find the classifier that minimizes the error with respect to the
        % distribution D
        SVMStruct = svmtrain(Xtr,Ytr,'BoxConstraint',D.*N);
        hypothesis(:,i) = svmclassify(SVMStruct,Xtr);
        testhypothesis(:,i) = svmclassify(SVMStruct,Xtst);        
        error = (abs(hypothesis(:,i) - Ytr).*0.5)'*D;
        % If error >= 0.5; stop
        if error >= 0.5
            break;
        end
        % set alpha
        alpha(i) = 0.5*log(1/error-1);
        % Update weights
        D = D.*exp(-alpha(i)*(Ytr.*hypothesis(:,i)))...
            ./sum(D.*exp(-alpha(i)*(Ytr.*hypothesis(:,i))));
        acc(k,i) = ((180-sum(abs(hypothesis*alpha...
        -Ytr))*0.5)*100)/180;
    end
    
    finalhypothesisTr(:,k) = sign(hypothesis*alpha);
    finalhypothesisTst(:,k) = sign(testhypothesis*alpha);
    trainAcc(1,k) = ((180-sum(abs(finalhypothesisTr(:,k)-Ytr))*0.5)...
        *100)/180;
    testAcc(1,k) = ((20-sum(abs(finalhypothesisTst(:,k)-Ytst))*0.5)...
        *100)/20;
    
    %Confusion matrix
    confMat(1,3,k) = size(find(Ytst==1),1);%p
    confMat(3,1,k) = size(find(finalhypothesisTst(:,k)==1),1);%pdash
    confMat(2,3,k) = size(find(Ytst==-1),1);%n
    confMat(3,2,k) = size(find(finalhypothesisTst(:,k)==-1),1);%ndash
    confMat(3,3,k) = confMat(1,3,k)+confMat(2,3,k);%N
    confMat(1,1,k) = size(intersect(find(Ytst==1),...
        find(finalhypothesisTst(:,k)==1)),1);%tp
    confMat(2,2,k) = size(intersect(find(Ytst==-1),...
        find(finalhypothesisTst(:,k)==-1)),1);%tn
    confMat(1,2,k) = confMat(1,3,k) - confMat(1,1,k);%fn
    confMat(2,1,k) = confMat(2,3,k) - confMat(2,2,k);%fp
end

%avg Confusion matrix
for i=1:3
    for j=1:3
        avgConfMat(i,j) = mean(confMat(i,j,:));
    end
end

%precision and recall
for k=1:kfold
    precision(1,k) = confMat(1,1,k)/confMat(3,1,k);
    precision(2,k) = confMat(2,2,k)/confMat(3,2,k);
    recall(1,k) = confMat(1,1,k)/confMat(1,3,k);
    recall(2,k) = confMat(2,2,k)/confMat(2,3,k);
end
avgPrecision(1,1) = mean(precision(1,:));
avgPrecision(1,2) = mean(precision(2,:));
avgRecall(1,1) = mean(recall(1,:));
avgRecall(1,2) = mean(recall(2,:));

%avg accuracy
avgacc = 0;
for t=1:T
   if(mean(acc(:,t))==0)
       break;
   end
   sz = size(find(acc(:,t)~=0),1);
   avgacc(t) = sum(acc(:,t))/sz;
end

%plot
figure;
plot(1:size(avgacc,2),avgacc);
xlabel('Number of Iterations')
ylabel('Accuracy')