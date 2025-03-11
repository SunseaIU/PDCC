clear;
clc;
close all;
warning off;

%=========读取源域目标域的数据
for i = 2:2
    [Xs,Xt,Ys,Yt] = load_data(i);
    Ys = Ys(:);  % 将Y转换为列向量
    Yt = Yt(:);  % 将Y转换为列向量
       
    %==========数据预处理
    [~,ns]=size(Xs);
    [~,nt]=size(Xt);
    % Xs=Xs-repmat(mean(Xs,2),[1,ns]);
    % Xt=Xt-repmat(mean(Xt,2),[1,nt]);
    % 
    % Xs=init_x(Xs);
    % Xt=init_x(Xt);
    
     
    %==========迭代更新的参数值,α，β，λ。还有投影矩阵的维度.
    
    % [Xs, Ys] = load_data();
    
    lambdalib=2.^(-10:10);
    alphalib=2.^(-10:10);
    betalib=2.^(-10:10);
    dimlib=[10: 10: 100];
    
    for t=1:length(dimlib)
        options.p=dimlib(t);
        for i1=1: length(alphalib)
             options.alpha=alphalib(i1);
             for i2=1 : length(lambdalib)
                 options.lambda=lambdalib(i2);
                 for i3=1:length(betalib)
                     options.beta=betalib(i3);
                     %===使用用b/A替换b*inv(A),用A\b替换inv(A)*b;
                      %[Ps_final,Pt_final,Ft_final,W_final,b_final,acc_best,obj_final]=RLRKT_update(Xs,Ys,Xt,Yt,options);
                     %===使用inv
                    [Ps_final,Pt_final,Ft_final,W_final,b_final,acc_best,obj_final]=RLRKT_update_useinv(Xs,Ys,Xt,Yt,options);
                     fprintf('dim=%d,alpha=%0.4f,lambda=%0.4f,beta=%0.4f,The  best acc=%0.4f\n',options.p,options.alpha,options.lambda,options.beta,acc_best);
                     if i1==1&&i2==1&&i3==1&&t==1
                         acc_finalbest=acc_best;
                         Ft_finalvalue=Ft_final;
                         Ps_finalvalue=Ps_final;
                         Pt_finalvalue=Pt_final;
                         b_finalvalue=b_final;
                         W_finalvalue=W_final;
                         obj_finalvalue=obj_final;
                     elseif acc_finalbest<acc_best
                         acc_finalbest=acc_best;
                         Ft_finalvalue=Ft_final;
                         Ps_finalvalue=Ps_final;
                         Pt_finalvalue=Pt_final;
                         b_finalvalue=b_final;
                         W_finalvalue=W_final;
                         obj_finalvalue=obj_final;
                     end
                 end
             end
        end
    end
end

fprintf('The final best acc=%0.4f\n', acc_finalbest);
savename=strcat('./GAKT/GAKT_P_sess1_sub1_sub2_eig1.mat');
save(savename,'acc_finalbest','Ft_final','Ps_final','Pt_final','b_final','W_final','obj_final');















