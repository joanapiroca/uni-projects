clearvars all;
clc;

A = rand([7,4]);
[U,S,V] = svd(A);
U
S
V
