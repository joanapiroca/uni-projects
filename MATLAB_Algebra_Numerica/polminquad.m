function [coefs, norm2Res] = polminquad(x,y,q,plt)
format long e;
x=x(:);
y=y(:);

%construeixo A
for i = 1:(q+1)
    A(:,i) = x.^(q+1-i);
end

%copio A per poder fer càlculs d'errors més tard:
Acopia = A;

%Descomposició QR (Gram-Smith modificat):
for i = 1:q+1
    R(i,i) = norm(A(:,i));
    A(:,i) = A(:,i) / R(i,i);
    
    for j = (i+1):(q+1)
        R(i,j) =  transpose(A(:,i))*A(:,j);
        A(:,j) = A(:,j) - R(i,j)*A(:,i);
    end
end

%Comprovo que Q és ortogonal, trec l'error per pantalla
fprintf('Error en el càlcul de Q és:')
norm( transpose(A)*A - eye(q+1), inf)

%Calculo i trec el vector de coeficients i la norma del residu
coefs = R\transpose(A)*y;
norm2Res = norm(Acopia*coefs-y, 2);

%si plt = 1 dibuixo la grafica amb els punts i el polinomi
if plt == 1
    figure(1)
    scatter(x,y, 'r', 'filled');
    hold on;
    
    xx = x(1)-0.01: 0.01: x(size(x))+0.01; %no se de que serveix
    yy = polyval(coefs,xx);
        
    plot(xx,yy,'b-', 'Linewidth', 1.25);
    hold off
end

end