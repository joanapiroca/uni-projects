function [x] = prova(x);

fileID = fopen('A.txt','r');
formatSpec = '%f ';
sizeA = [20 10]
A = fscanf(fileID,formatSpec, sizeA)
fclose(fileID);


end