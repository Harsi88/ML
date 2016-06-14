load('recvstalkmini.mat');
tr=trdata;
ts = tsdata;
trl = trlabel;
tsl = tslabel;


[n,~]=size(ts);
[m,~]=size(tr);

rand('seed',15);
c = [1 5 10 20];
for v = 1:4
for l=1:10
    
    testhypothesis = [];
    hypothesis = [];
    b = [];
    b0 = [];
    nhypo = [];
    e = [];
    Xtr = [];
    Xts = [];
    
    
T = 500;
r = randsample(size(tr,1),floor(size(tr,1)*c(v)/100));
s = randsample(size(ts,1),floor(size(ts,1)*c(v)/100));
w = ones(floor(n*c(v)/100)+floor(m*c(v)/100),1);
        Xtr = tr(r,:);
        Ytr = trl(r,:);
        Xtrt = tr;
        Ytrt = trl;
        j=1;
        count = 0;
        for k =1:size(tr,1)
            if(k == r(j))
                Xtrt(k-count,:)=[];
                Ytrt(k-count,:)=[];
                j=j+1;
                count = count+1;
            end
        end
        Xts = ts(s,:);
        Yts = tsl(s,:);
        Xtst = ts;
        Ytst = tsl;
        j=1;
        count = 0;
        for k =1:size(ts,1)
            if(k == s(j))
                Xtst(k-count,:)=[];
                Ytst(k-count,:)=[];
                j=j+1;
                count = count+1;
            end
        end
for i =1:T
        p = w./sum(w);
        alpha = mysvmseprabledual([Xtr;Xts],[Ytr;Yts],p);
        temp1 = ((alpha.*[Ytr;Yts])'*[Xtr;Xts]);
        temp2 = mean(([Xtr;Xts]*temp1')-[Ytr;Yts]);
        hypothesis(:,i) = [Xtr;Xts]*temp1'+temp2;
        hypothesis = hypothesis./abs(hypothesis);
        testhypothesis(:,i) = [Xtrt;Xtst]*temp1'+temp2;
        e(i) = (w(size(Xtr,1)+1:size(Xtr,1)+size(Xts,1))'*abs(hypothesis(size(Xtr,1)+1:size(Xtr,1)+size(Xts,1),i)-Yts))/sum(w(size(Xtr,1)+1:size(Xtr,1)+size(Xts,1)));
        b(i) = e(i)/(1-e(i));
        b0(i) = 1/(1+(2*log(size(Xtr,1))/T).^.5);
        if(e(i)<.5)
            break;
        end
        
        w(1:size(Xtr,1),1)=w(1:size(Xtr,1),1).*(b0(i).^abs(hypothesis(1:size(Xtr,1),i)-Ytr));
        w(size(Xtr,1)+1:size(Xtr,1)+size(Xts,1),1)=w(size(Xtr,1)+1:size(Xtr,1)+size(Xts,1),1).*(b(i).^(-1*abs(hypothesis(size(Xtr,1)+1:size(Xtr,1)+size(Xts,1),i)-Yts)));
        
end
        testhypothesis = testhypothesis./abs(testhypothesis);
        num = size(testhypothesis,2);
        num2 = size(testhypothesis,1);
        for i = 1:(num2)
            bnew = b(1,floor(num/2)+1:num);
            tnew = testhypothesis(i,floor(num/2)+1:num);
        nhypo(i) = prod(bnew.^(-1*(tnew)))>prod(bnew.^-0.5);
        
        end
        nhypo = nhypo*2-1;
        fi = nhypo' - [Ytrt;Ytst];
        zoom = find(fi);
        accuracy(v,l) = (num2-size(zoom,1))/num2;
end
end
accuracy