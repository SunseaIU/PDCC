function [Xs] = init_x(Xs)
%INIT_X 此处显示有关此函数的摘要
%   此处显示详细说明

    Xs = Xs';   % Xs: n-by-d
    Xs = Xs ./ repmat(sum(Xs,2),1,size(Xs,2));  % d-by-d
    Xs = zscore(Xs);
    Xs = normr(Xs);
    Xs= Xs';
end

