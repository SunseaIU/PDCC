clear;clc;close all;

%读取源域目标域的数据
Xs=load('C:\Users\Liu\Desktop\GEKT\SEED4\subject1\fea_session1_subject1.mat');
Xs=Xs.fea;
Xt=load('C:\Users\Liu\Desktop\GEKT\SEED4\subject1\fea_session2_subject1.mat');
Xt=Xt.fea;
Ys=load('C:\Users\Liu\Desktop\GEKT\SEED4\subject1\gnd_session1.mat');
Ys=Ys.gnd;
Yt=load('C:\Users\Liu\Desktop\GEKT\SEED4\subject1\gnd_session2.mat');
Yt=Yt.gnd;



[~,ns]=size(Xs);%返回向量[d,ns]
[~,nt]=size(Xt);
c=max(Ys);%返回列数，即类
Ys=onehot(Ys,c);

%数据预处理
Xs=Xs-repmat(mean(Xs,2),[1,ns]);
Xt=Xt-repmat(mean(Xt,2),[1,nt]);

%迭代更新的参数值
lambdalib=10.^(-3:3);
alphalib=10.^(-3:3);
betalib=10.^(-3:3);
dimlib=[10: 10: 100];

%目标域标签的初始化
Ft_init=ones(nt,c)./c;

Y= [Ys;Ft_init]; %初始化的目标域标签和已知的源域标签拼接
%投影矩阵的维度，目标函数的三个参数，alpha，beta，lambda。
for t=1:length(dimlib)
    options.p=dimlib(t);
    for i1=1: length(alphalib)
         options.alpha=alphalib(i1);
         for i2=1 : length(betalib)
             options.beta=betalib(i2);
             for i3=1:length(lambdalib)
                 options.lambda=lambdalib(i3);
                 
                 %每组参数进入目标函数后的迭代----
                 [Ps,Pt,Ft,b,W,acc,obj]=demo1(Xs,Ys,Xt,Yt,Ft_init,Y,options);
                 fprintf('dim=%d,lambda=%.4f,beta=%0.4f,alpha=%0.4f,acc=%0.4f \n',options.p,options.lambda,options.beta,options.alpha,acc);
                 if i1==1&&i2==1&&i3==1&&t==1
                     acc_best=acc;
                     Obj=obj;
                     Ft_pre=Ft;
                     Ps_pre=Ps;
                     Pt_pre=Pt;
                     b_pre=b;
                     W_pre=W;
                     option.dim=options.p;
                     option.lam=options.lambda;
                     option.beta=options.beta;
                     option.al=options.alpha;
                 elseif acc_best<acc
                     acc_best=acc;
                     Obj=obj;
                     Ft_pre=Ft;
                     Ps_pre=Ps;
                     Pt_pre=Pt;
                     b_pre=b;
                     W_pre=W;
                     option.dim=options.p;
                     option.lam=options.lambda;
                     option.beta=options.beta;
                     option.al=options.alpha;
                 end
                     
             end
         end
    end
end

fprintf('The best acc=%0.4f\n',acc_best);                 
                 
        










