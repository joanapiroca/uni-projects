function [x,rho,res,iter] = gaussS(A,b,x0,nmax,prec)
format long e;
x0=x0(:);
L = tril(A, -1);
D = diag(diag(A));

%Comprovo que A no té zeros a la diagonal
tol=1.0e-10;
if min(abs(diag(D)))<tol 
    warning('error: values on the diagonal with abs. val < %e',tol)
end

BGS = eye(size(A)) - (L+D)\A;
CGS = (D+L)\b;

rho = max(abs(eig(BGS)));
if rho >= 1 
    warning('El mètode no convergeix (no es compleix "rho<1").')
end

x = x0;
iter = 0;
test = ( norm(b-A*x,inf)/norm(b-A*x0,inf) );
while iter <= nmax & test >= prec
    x = BGS*x + CGS;
    test = ( norm(b-A*x,inf)/norm(b-A*x0,inf) );
    iter = iter + 1;
end
res = norm(A*x-b, inf);

end
