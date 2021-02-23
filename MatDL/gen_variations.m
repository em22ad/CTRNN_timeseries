function Y=gen_variations(X,every,far)
%far=0 means copy value from just 1 place to right of left, far=1 means copy from within 10 values from right or left  
Y=X;
for i=1:size(X,2)
    if (every == 0)
        continue;
    end
    if (mod(i,every) == 0)
        %X(1,i)=(extent*(X(1,i+1)-X(1,i))+X(1,i));
        idx=(int8(rand()*size(X,2))*far);
        if (rand() > 0.5)
            idx=-idx;
        end
        
        if (idx+i <=0)
            idx=1;
        end
        
        if idx+i > size(X,2)
            idx=size(X,2)-i;
        end
        
        Y(1,i)=X(1,i+idx);
    end
end