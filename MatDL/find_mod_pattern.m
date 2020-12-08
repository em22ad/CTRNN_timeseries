function [idxs,pattern_o]=find_mod_pattern(X,T,every,far,num_mods)
%idxs=[];
pattern_o=return_pattern(X);
idxs=strfind(T,pattern_o);
for i=1:num_mods
    %Y=gen_variations(X,every,far);
    %pattern=return_pattern(Y);
    %idx_loc=strfind(T,pattern);
    
    Y=[];
    while size(Y,2) ~= size(pattern_o,2)
        Y=str_mutate(pattern_o,far);
    end
        
    idx_loc=strfind(T,Y);
    if size(idx_loc,2) > 0
        idxs=[idxs idx_loc];
    end
end