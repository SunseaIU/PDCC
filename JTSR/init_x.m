function [Xs] = init_x(Xs)
%INIT_X �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��

    Xs = Xs';   % Xs: n-by-d
    Xs = Xs ./ repmat(sum(Xs,2),1,size(Xs,2));  % d-by-d
    Xs = zscore(Xs);
    Xs = normr(Xs);
    Xs= Xs';
end

