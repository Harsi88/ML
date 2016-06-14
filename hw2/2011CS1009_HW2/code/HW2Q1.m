data1 = normrnd(2,0.2,1,10);
data2 = normrnd(3,0.2,1,10);
data3 = normrnd(4,0.2,1,10);

hold on;
plot(data1,0,'bo');
plot(data2,0,'go');
plot(data3,0,'ro');

testSample = 1:0.1:5;
D = 1;
Var1 = var(data1);
Var2 = var(data2);
Var3 = var(data3);
Mean1 = mean(data1);
Mean2 = mean(data2);
Mean3 = mean(data3);
mat1 = testSample - Mean1;
mat2 = testSample - Mean2;
mat3 = testSample - Mean3;

h1 = zeros(1,41);
h2 = zeros(1,41);
h3 = zeros(1,41);

for i = 1:41
    temp = ( 1 / ( ( ( 2*pi )^( D/2 ) ) * Var1^0.5 ) );
    h1(1,i) = temp*exp((-1/(2*Var1))*mat1(1,i)^2);
    temp = ( 1 / ( ( ( 2*pi )^( D/2 ) ) * Var2^0.5 ) );
    h2(1,i) = temp*exp((-1/(2*Var1))*mat2(1,i)^2);
    temp = ( 1 / ( ( ( 2*pi )^( D/2 ) ) * Var3^0.5 ) );
    h3(1,i) = temp*exp((-1/(2*Var1))*mat3(1,i)^2);
end

plot(testSample, h1, 'b')
plot(testSample, h2, 'g')
plot(testSample, h3, 'r')

h11 = h1./(h1+h2+h3);
h22 = h2./(h1+h2+h3);
h33 = h3./(h1+h2+h3);
plot(testSample, h11, 'b--')
plot(testSample, h22, 'g--')
plot(testSample, h33, 'r--')
title('plotting likelihood and posterior probability')

hold off
ylim([-1 Inf]);

Var = (Var1+Var2+Var3)/27;

d1 = testSample*Mean1/Var - 0.5*Mean1*Mean1/Var + log(1/3);
d2 = testSample*Mean2/Var - 0.5*Mean2*Mean2/Var + log(1/3);
d3 = testSample*Mean3/Var - 0.5*Mean3*Mean3/Var + log(1/3);

[ intx1, inty1 ] = polyxpoly(testSample,d1,testSample,d2);
[ intx2, inty2 ] = polyxpoly(testSample,d3,testSample,d2);

figure;

hold on
plot(data1,-500,'bo');
plot(data2,-500,'go');
plot(data3,-500,'ro');
plot(testSample, d1, 'b')
plot(testSample, d2, 'g')
plot(testSample, d3, 'r')
plot(intx1,0,'k')
yL = get(gca,'YLim');
line([intx1 intx1],yL,'Color','k');
line([intx2 intx2],yL,'Color','k');
for i = 1:41
    if testSample(1,i) < intx1
        plot(testSample(1,i),0,'b*')
    elseif testSample(1,i) < intx2
        plot(testSample(1,i),0,'g*')
    else
        plot(testSample(1,i),0,'r*')
    end
end
title('Linear Discriminant Functions')
hold off