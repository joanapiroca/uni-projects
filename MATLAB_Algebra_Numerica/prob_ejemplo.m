format long;

clearvars all;  %limpia todas las variables
clc;    %limpia todo el texto de la comanda de windows

x = [0.; 0.55; 1.25; 2.15;3.17];
y = [-2.1; 0.25; 1.56; -0.13; -1.4];

%plot example
figure(1)
scatter(x,y, 'r', 'filled');
hold on;

%plotting a curve

%def the function
xx = -0.01:0.01:3.25;
fun = @(t) (1-exp(-t/3).*sin(10*t));
zz = fun(xx); 

figure(2)
plot(xx, zz, 'm-.', 'Linewidth', 1.25);


%bucle for
N=length(x)
for j=1:N
    sum(j) = x(j) + y(j);
end

%if...else
if(length(x) ~= length(y))
    fprintf('Warning: different lengths \n');
else
    fprintf('Ok. Their length is %d\n', length(x));
    
    %creo la matriu A
    A = [x.^4,x.^3,x.^2,x.^1,x.^0];
    
    if(det(A) < 10^-10)
        error('Break');
        fprintf('Matrix is singular');
    
    else
        coeff = A\y;
        
        xx = x(1)-0.005: 0.01: x(5)+0.05;
        yy = polyval(coeff,xx);
        
        figure(1)
        plot(xx,yy,'r-', 'Linewidth', 1.25);
        hold off
    end
    
    
end

%calling a function
[vsum,normdos]=sumavec(x,y);






