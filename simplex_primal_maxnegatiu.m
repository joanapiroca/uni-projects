function[VB, XB, VNB, R, Z] = simplex_primal_maxnegatiu(c,A,b)
    fileID = fopen('resultats_maxnegatiu.txt','w');

    m = size(A,1);
    n = size(A,2);
    fase = 1; %Indica a quina fase em trobo
    fi = false; %quan sigui true atura l'algorisme del simplex

    %guardo les dades per la fase II
    ccopia = c;
    Acopia = A;
    bcopia = b;

    %Dades inicials de la fase I
    A = [A eye(m)];
    c = [zeros(1,n) ones(1,m)];
    b = b;
    vb= n+1:n+m;
    vnb = 1:n;
    x = [zeros(1,n) b];
    z = sum(b);
    iter = 1;

    fprintf(fileID,'Inici simplex primal amb regla del cost reduït més negatiu\n');
    fprintf(fileID,'   Fase I\n');
    %Inicio l'algoisme del simplex
    while(fi == false) 
        %calculo el vector de costos reduïts
        Binv = A(:,vb)^(-1);
        r = c(:,vnb) - c(:,vb)*Binv*A(:,vnb);

        xb = x(:,vb);
        xb = xb(:);

        if r >= 0
            %Si estem a la fase IIhem trobat l'òptim
            if fase == 2
                fi = true;

            %Si estic a la fase I estudio la factibilitat del problema
            else
                %si z == 0el problema és factible
                %Actualitzo variables i passo a la fase II
                if abs(z) < 10e-9
                        fprintf(fileID,'      Solució òptima trobada, iteració %d\n',iter);
                        fprintf(fileID,'   Fase II\n');

                        x = x(:,1:n);
                        z = ccopia*(x(:));
                        k = 1;
                        for i = 1 : m
                            if (vb(i) > n) 
                                vb(i) = vnb(k);
                                k = k + 1;
                            end
                        end
                        vnb = vnb(:,k:n-m);
                        fase = 2;
                        A = Acopia; 
                        b = bcopia; 
                        c = ccopia;
                        %Reinició el contador de iteracions
                        iter = 1;

                    %El problema és infactible
                    else 
                        error(fileID,'Problema infactible\n')
                        para = false;
                end
            end

            %No hem trobat l'òptim i el problema és factible, segueixo endavant
            %amb l'algorisme
            else
            %algún element de r és negatiu, encara no he trobat l'òptim
            %escullo q (vnb) amb la regla de Bland, 
            %guardo la j (posició de la vnb q al vector vnb)
            j = 1;
            while ( r(j) ~= min(r))
                j = j+1;
            end
            q = vnb(j);

            %Calculo la direccio basica descens associada a q
            db = -Binv*A(:,q); 
            %Estudio si el problema és il·limitat
            if db >= 0
                error('Problema il·limitat') ; 
                para = false;
            %Si el problema no és il·limitat continuo endavant amb l'algorisme
            else 
                %Calculo la variable sortint p i la passa màxima theta*
                p = 0;
                for i = 1 : size(db)
                    if db(i) < 0
                       if p == 0
                           theta = -xb(i)/db(i);
                           p = i;
                       else 
                           aux = -xb(i)/db(i);
                           if min(theta, aux) == aux
                               theta = aux;
                               p = i;
                           end   
                       end
                    end
                end

                rq = r(j);
                z = z + theta*rq; 

                %Escric les dades de la iteració
                formatSpec = '      Iteració  %3d : q = %3d, rq = %8.3f, B(p) = %3d, theta*= %8.3f, z = %8.3f \n';
                fprintf(fileID,formatSpec,iter,q,rq,vb(p),theta,z);

                %ACTUALITZO VARIABLES I CANVIO DE BASE;
                xb = xb + theta*db;
                x(:,vb) = xb;
                x(q) = theta;
                vnb(j) = vb(p);
                vb(p) = q; 
                vnb = sort(vnb);
                xb = x(:,vb);            
                iter = iter + 1;
                %Torno a l'inici del bucle fins trobar l'òptim, infactibilitat
                %o que el problema és il·limitat
            end 
         end    
    end

    q = 0;
    rq = 0;
    vb(p) = 0; 
    theta = 0;

        formatSpec = '      Iteració  %3d : q = %3d, rq = %8.3f, B(p) = %3d, theta*= %8.3f, z = %8.3f \n';
        fprintf(fileID,formatSpec,iter,q,rq,vb(p),theta,z);
    fprintf(fileID,'      Solució òptima trobada, iteració %d, z = %8.6f\n',iter,z);
    %comprovo si hi ha degeneració
    deg = false;
    i = 1;
    while ( deg == false & i <= size(xb))
        if xb(i) == 0
            fprintf(fileID,'degenerat');
            deg = true;
        end
        i = i+1;    
    end

    fprintf(fileID,'Fi simplex primal\n');
    fclose(fileID);
    VB = vb;
    XB = xb;
    VNB = vnb;
    R = r;
    Z = z;
    X = x;

end