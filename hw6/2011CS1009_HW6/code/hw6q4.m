close all; clear all;

load('hw6data.txt');

X = hw6data';

[N,D] = size(X);

DM = zeros(N,N);

for i=1:N
    for j=1:N
        DM(i,j) = norm(X(i,:)-X(j,:));
    end
end

DG(:,:) = DM;

% for k=1:N
%     for i=1:N
%         for j=1:N
%             DG(i,j) = min(DG(i,j),DG(i,k)+DG(k,j));
%         end
%     end
% end

S = zeros(N,N);
for i=1:N
    for j=1:N
        S(i,j) = -0.5*(DG(i,j)^2 - mean(DG(i,:))^2 - mean(DG(j,:))^2);
    end
end

[U, A] = eigs(S);

PCA = U*(A.^0.5);
