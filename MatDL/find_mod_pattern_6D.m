function [idxs,pattern_o]=find_mod_pattern_6D(X,T,every,far,num_mods)
%idxs=[];
pattern_o(1,:)=return_pattern(X(1,:));
pattern_o(2,:)=return_pattern(X(2,:));
pattern_o(3,:)=return_pattern(X(3,:));

idxs1=strfind(T(1,:),pattern_o(1,:));
idxs2=strfind(T(2,:),pattern_o(2,:));
idxs3=strfind(T(3,:),pattern_o(3,:));

idxs=intersect(intersect(idxs1,idxs2),idxs3);

for i=1:num_mods
    %Y=gen_variations(X,every,far);
    %pattern=return_pattern(Y);
    %idx_loc=strfind(T,pattern);
    
    Y1=[];
    Y2=[];
    Y3=[];
    
    while size(Y1,2) ~= size(pattern_o(1,:),2)
        Y1=str_mutate(pattern_o(1,:),far);
    end

    while size(Y2,2) ~= size(pattern_o(2,:),2)
        Y2=str_mutate(pattern_o(2,:),far);
    end

    while size(Y3,2) ~= size(pattern_o(3,:),2)
        Y3=str_mutate(pattern_o(3,:),far);
    end

    idx_loc1=strfind(T(1,:),Y1);
    idx_loc2=strfind(T(2,:),Y2);
    idx_loc3=strfind(T(3,:),Y3);
    
    idx_loc=intersect(intersect(idx_loc1,idx_loc2),idx_loc3);
    
    if size(idx_loc,2) > 0
        idxs=[idxs idx_loc];
    end
end