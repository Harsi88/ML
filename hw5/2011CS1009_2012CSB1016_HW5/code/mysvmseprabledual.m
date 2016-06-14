function [alpha] = mysvmseprabledual(X, Y,D)
[n,~]=size(X);
f = -ones(n,1);
H = (Y*Y').*(X*X');
A = [];
b = [];
Aeq = Y';
beq = 0;
lb = zeros(n,1);
ub = 0.1.*D;
opts = optimset('algorithm','interior-point-convex');
alpha = quadprog(H,f,A,b,Aeq,beq,lb,ub,[],opts);
end    