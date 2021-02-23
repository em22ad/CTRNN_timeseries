function y=str_mutate(pattern,how_close)
    %t= '001001';
    b=num2str(pattern);
    t = strrep(b,' ','');
    numRands=length(t);
    tlength=numRands;
    source=t(ceil(rand(1,tlength)*numRands));
    fitval = fitness(source, t);
    i = 0;
    while i < how_close
         i = i + 1;
         m = mutate(source);
         fitval_m = fitness(m, t);
         if fitval_m < fitval
            fitval = fitval_m;
            source = m;
            %fprintf('%5i %5i %14s', i, fitval_m, m);
         end
         if fitval == 0
             break
         end
    end
    d=cellstr(reshape(m,1,[])');
    y=str2num(cell2mat(d));
    y=y';
    y(y>1)=1;
    y(y<0)=0;
end 
function fitval = fitness(source, t)
     fitval = 0;
     for i = 1 : length(source)
fitval = fitval + (double(t(i)) - double(source(i))) ^ 2;
     end
end
function parts = mutate(source)
     parts = source;
     charpos = randi(length(source));
     parts(charpos) = char(double(parts(charpos)) + (randi(3)-2));
end