format long; % això fa que la sortida sigui amb més decimals

% llegim d'un fitxer
A = xlsread('world_population_1800.xls');
year=A(:,1);
popul=A(:,2);
n=length(year);  % longitud del vector year

% fem un primer dibuix
figure(1)
scatter(year,popul,'r','filled'); % dibuixa punts
hold on; % això permet continuar dibuixant sobre aquesta figura
         % més endavant. Per acabar: hold off

%busquem aproximació tipus p = A exp(b*y)
% z=log(popul) , z = alfa + By
z = log(popul);

%construim la matriu M (=A)
M = [ones(n,1), year];
Mcopia = M;
%primer pas de Gram-Smith modificat
R(1,1) = norm(M(:,1),2);
M(:,1) = M(:,1)/R(1,1);
R(1,2) = dot(M(:,1),M(:,2));
M(:,2) = M(:,2) - R(1,2)*M(:,1);

%segon pas
%normalitzem columna 2
R(2,2) = norm(M(:,2),2);
M(:,2) = M(:,2)/R(2,2)
%Q=M
Q=M;
%comprovem que Q es ortogonal, trec Qt*Q i m'ha de treure la identitat
transpose(Q)*Q
%Resolem les equacions normals
sol = R\(transpose(Q)*z);
alpha = sol(1);
B =sol(2)

%dibuixar a sobre de la figura 1;
%comparem punts i figura
figure(1)
scatter(year,z,'r', 'filled');
yy =(year(1)+0.01):0.01:(year(n)+0.01);
zaprox = alpha+B*yy;
plot(yy,zaprox, '-');

