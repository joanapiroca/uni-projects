A = readmatrix('matriu-A.dat');
n = sqrt(length(A));
A = reshape(A,n,n);

D = diag(diag(A));
%tril (lower) triu (upper) dona la matriutrangular inf o sup i -1 vol dir
%una diagonal per sota de la diagonal principal;
L = tril(A, -1);
U = triu(A, +1);
omega = 0.25;
Bomega = inv(D+omega*L)*((1-omega)*D-omega*U);
rho = max(abs(eig(Bomega)));


