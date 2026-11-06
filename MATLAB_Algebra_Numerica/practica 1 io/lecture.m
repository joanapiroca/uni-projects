function [A, b, c] = lecture(nom_document);
  
  fileID = fopen(nom_document, 'r');

  if (fileID == -1) 
    error ("No s'ha pogut llegir l'arxiu");
  endif

  mida_C = fscanf(fileID, '%d', 2);
  c = fscanf(fileID, '%f', mida_C);
  c = c';
 
  mida_A1 = fscanf(fileID, '%d', 2);
  A1 = fscanf(fileID, '%f', mida_A1);
  A1 = A1';
  mida_A2 = fscanf(fileID, '%d', 2);
  A2 = fscanf(fileID, '%f', mida_A2);
  A2 = A2';
  A = [A1, A2];
  
  mida_b = fscanf(fileID, '%d', 2);
  b = fscanf(fileID, '%f', mida_b);
  b = b'; 
 
  fclose(fileID);
    
end