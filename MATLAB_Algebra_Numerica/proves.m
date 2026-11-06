function [VB, XB, VNB, R, Z, X] = proves(c, A, b)

fileID = fopen('resultat.txt','w');
    tol = 10e-7;
    n = size(A,2);
    m = size(A,1);
    eye(m);
    %variables que necessitarem en la fase II si s'escau
    ccopia = c; 
    Acopia = A; 
    bcopia = b;
    %variables de la fase I
    c = [zeros(1,n) ones(1,m)];
    A = [A eye(m)];
    b = b;
    vb = n+1:n+m;
    vn = 1:n;
    x = [zeros(1,n) b];
    z = sum(b);
    it = 1;
    fase = 1;
    
    para = true;
    
    fprintf(fileID,'Inici simplex primal amb regla de Bland\n');
    fprintf(fileID,'   Fase I\n');
    while (para == true) 
    %calculem vector de costos reduïts
        r = c(:,vn) - c(:,vb)*(A(:,vb)^(-1))*A(:,vn);

        xb = x(:,vb);
        xb = xb(:);

        %hem trobat l'òptim
        if r >= 0 
            % si està a la fase 2 hem trobat el resultat que buscàvem
            if fase == 2 
                para = false;
            %si està a la fase I mirem si és factible i passem a fase II o
            %infactible
            else 
                if abs(z) < tol
                    fprintf(fileID,'      Solució òptima trobada, iteració %d\n',it);
                    fprintf(fileID,'   Fase II\n');
                     it = it+1;
                    x = x(:,1:n);
                    z = ccopia*(x(:));
                    k = 1;
                    for i = 1 : m
                        if (vb(i) > n) 
                            vb(i) = vn(k);
                            k = k + 1;
                        end
                    end
                    vn = vn(:,k:n-m);
                    fase = 2;
                    A = Acopia; 
                    b = bcopia; 
                    c = ccopia;
                else 
                    error(fileID,'Problema infactible\n');
                    para = false;
                end
            end     

        %sinó seguim buscant
        else
            %calculem la q, vnb que entrarà a la base
            %Regla de Bland
            i = 1; 
            while (r(i) >= 0) 
                i = i+1;         
            end
            q = vn(i);
            %calculem direcció bàsica associada
            db = -A(:,vb)^(-1)*A(:,q);
            if db >= 0 
                error(fileID,'Problema il·limitat\n');
                para = false;
            else  
                %calculem la p i la theta
                p = 0;
                for j = 1 : size(db)
                    if db(j) < 0
                        if p == 0
                            theta = -xb(j)/db(j);
                            p = j;          
                        else
                            if min(theta, -xb(j)/db(j)) == -xb(j)/db(j)
                                theta = -xb(j)/db(j);
                                p = j;    
                            end
                        end

                    end
                end  
            rq = r(i);
            z = z + theta*rq;
            formatSpec = '      Iteració  %3d : q = %3d, rq = %8.3f, B(p) = %3d, theta*= %8.3f, z = %8.3f \n';
            fprintf(fileID,formatSpec,it,q,rq,vb(p),theta,z);

            %actualitzem dades
            xb = xb + theta*db;
            x(:,vb) = xb;
            x(q) = theta;
            it = it +1;
            vn(i) = vb(p);
            vb(p) = q; 
            %ordenem variables no bàsiques, ja que ens facilita buscar la q
            vn = sort(vn);
            xb = x(:,vb);
            end
            
        end    
        %tornem a cridar la funció fins a trobar l'òptim o veure que no hi ha sol·lució
    end
    fprintf(fileID,'      Solució òptima trobada, iteració %d, z = %8.6f\n',it,z);
    fprintf(fileID,'Fi simplex primal\n');
    fclose(fileID);
    VB = vb; 
    XB = xb; 
    VNB = vn;
    R = r; 
    Z = z;
    X = x;
end
    