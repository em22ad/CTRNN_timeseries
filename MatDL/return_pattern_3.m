function pattern=return_pattern_3(X)
pattern=[];
for i=1:size(X,2)-1
    if (X(i) > X(i+1)-0.1*X(i)) && ((X(i) < X(i+1)+0.1*X(i)))
        pattern=[pattern 2];
    elseif X(i) < X(i+1) 
        pattern=[pattern 1];
    else%if X(i) > X(i+1) 
        pattern=[pattern 0];
    end
end