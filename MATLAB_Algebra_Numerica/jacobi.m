function [x,rho,res,iter] = jacobi(A,b,x0,nmax,prec)
format long e;
x0=x0(:);
D = diag(diag(A));
%Comprovo que A no té zeros a la diagonal
tol=1.0e-10;
if min(abs(diag(D)))< tol 
    warning('error: values on the diagonal with abs. val < %e')
end

BJ = eye(size(A)) - D\A; %IDENTITAT?????
CJ = D\b;

rho = max(abs(eig(BJ))); 
if rho >= 1 
    warning('El mètode no convergeix (no es compleix "rho<1").')
end

x = x0;
iter = 0;
test = ( norm(b-A*x,inf)/norm(b-A*x0,inf) );
while iter <= nmax & test >= prec
    x = BJ*x + CJ;
    iter = iter + 1;
    test = ( norm(b-A*x,inf)/norm(b-A*x0,inf) );
end
res =  norm(A*x-b,inf);
end
 