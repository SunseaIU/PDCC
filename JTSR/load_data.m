function [Xs, Xt, Ys, Yt] = load_data(i)
% LOAD_DATA �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
S = load(['.\data\001-2014selectnew\001-2014subject_' num2str(i) '_select.mat']);
Xs = S.x';
Ys = S.y + 1;

T = load(['.\data\data_align\MI2_align\subject_' num2str(i) '_aligned.mat']);
Xt = T.tar_fea';
Yt = T.tar_label' + 1;
end