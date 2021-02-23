function pattern=return_pattern(X)
pattern=[];
for i=1:size(X,2)-1
    if X(i) > X(i+1)
        pattern=[pattern 0];
    else%if X(i) < X(i+1) 
        pattern=[pattern 1];
    end
end