clearvars all;  %limpia todas las variables
clc;    %limpia todo el texto de la comanda de windows

A = [2,5,2;5,2,5;5,5,2];
b = [12;12;12];

D = diag(diag(A));
BJ = inv(D)*(D-A);
cJ = inv(D)*b;
rhoBJ = max(abs(eig(BJ)));

x(:,1) = zeros(3,1)
for i = 2:11
    x(:,i) = BJ*x(:,i-1) + cJ;
end

l =zeros(1,11);
for i = 2:11;
    l(1,i) = log( norm(x(:,i), 2) );
end
l_ap = zeros(1,11);
for i = 2:11
    l_ap(1,i) = log(rhoBJ^i * norm(x(:,2), 2) );
end

figure(1)
scatter(2:11, l(:,2:11), 'r', 'filled');
hold on;
scatter(2:11, l_ap(:,2:11), 'b', 'filled');
t = (1-0.01):0.01:(11+0.01);
caca = log(norm(x(:,2), 2));
l_ap2 = t.*log(rhoBJ)+caca;
plot(t, l_ap2, '-b');







