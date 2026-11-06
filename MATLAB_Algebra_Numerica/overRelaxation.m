function [x,rho,res,iter] = overRelaxation(A,b,x0,w,nmax,prec)
format long e;
x0=x0(:);
L = tril(A, -1);
U = triu(A, +1);
D = diag(diag(A));

%Comprovo que A no té zeros a la diagonal
tol=1.0e-10;
if min(abs(diag(D)))<tol 
    warning('error: values on the diagonal with abs. val < %e')
end

%comprovo que w està entre 0 i 2
if w >=2 | w <= 0
     warning('w no pertany a l"interval (0,2).')
end

BS = (D+w*L)\((1-w)*D-w*U);
CS = w*((D+w*L)\b);
rho = max(abs(eig(BS)));

if rho >= 1 
    error('El mètode no convergeix (no es compleix "rho<1").')
end

x = x0;
iter = 0;
test = ( norm(b-A*x,inf)/norm(b-A*x0,inf) );
while iter < nmax & test >= prec
    x = BS*x + CS;
    test = ( norm(b-A*x,inf)/norm(b-A*x0,inf) );
    iter = iter + 1;
end
res = norm(A*x-b, inf);

end