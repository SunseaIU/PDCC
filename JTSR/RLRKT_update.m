function   [Ps_final,Pt_final,Ft_final,W_final,b_final,acc_best,obj_final]=RLRKT_update(Xs,Ys,Xt,Yt,options)
%RLRKT_UPDATE 传入的参数：Xs,Ys,Xt,Yt,以及options中的α，β，λ，投影矩阵P的维度.

%=======取出这次迭代的参数
p=options.p;
alpha=options.alpha;
beta=options.beta;
lambda=options.lambda;

%======数据的初始初始维度d.
d=size(Xs,1);
%======源域，目标域的样本个数，ns/nt
[~,ns]=size(Xs);
[~,nt]=size(Xt);


%=====总样本个数，n
n=ns+nt;

%======标签总数：C
c=max(Ys);

%======将源域的标签进行onehot编码：Ys_onehot
Ys_onehot=onehot(Ys,c);

%=====>> Yss=[1,Ys],Ns=diag(1/ns_i),Nss=diag(1/ns,Ns)
%=====>> %有提示：用b/A替换b*inv(A),用A\b替换inv(A)*b。
Yss=[ones(ns,1),Ys_onehot];
Ns=diag(sum(Ys_onehot));
Nss=diag([ns,sum(Ys_onehot)]);

%=======目标域标签的初始化。
Ft_init=ones(nt,c)./c;
Ft=Ft_init;

%=========初始化的目标域标签和已知的源域标签拼接:Y
Y= [Ys_onehot;Ft_init];

%=========源域和目标域：X
X=[Xs,zeros(d,nt);zeros(d,ns),Xt];


%=========中心矩阵
H = eye(n)-1/n * ones(n,n);

%=========初始化投影矩阵P,使用PCA。
p_dim=[];
p_dim.ReducedDim=p;
[P,~]=PCA(X',p_dim);
Ps=P(1:d,:);
Pt=P(d+1:2*d,:);
%=========初始化W,使用PCA。
W_x=X'*P;
w_dim=[];
w_dim.ReducedDim=c;
[W,~]=PCA(W_x,w_dim);


%======进入循环迭代之前初始化的参数有：Ft--->Ft_init---->Y= [Ys_onehot;Ft_init].W,P.
NITER=20;
for iter=1:NITER
    %======根据求解结果更新b
    b=1/n*(Y'*ones(n,1)-W'*P'*X*ones(n,1));
    %======根据求解结果，更新W
    W=(P'*X*H*X'*P+beta*eye(options.p))\(P'*X*H*Y);
    
    %======根据求解结果更新Ft
    Nt=diag(sum(Ft));
    
    %=====>> %有提示：用b/A替换b*inv(A),用A\b替换inv(A)*b。
    Zs=Xt'*Pt*Ps'*Xs*Ys_onehot/(Ns*Nt);
    Zt=Xt'*(Pt*Pt')*Xt*Ft/(Nt*Nt);
    %====>> Zt=Zt(+)-Zt(-),Zs=Zs(+)-Zs(-).
    Ztz = Zt;
    Ztz(Ztz<0)=0;%负值全部更换为0
    Zsz = Zs;
    Zsz(Zsz<0)=0;%负值全部更换为0
    Ztf = Zt;
    Ztf(Ztf>0)=0;%正值全部替换为0
    Zsf = Zs;
    Zsf(Zsf>0)=0;%正值全部替换为0
    Fw =ones(nt,1)*ones(c,1)'+Xt'*Pt*W;
    Fd =lambda*Ft*ones(c,1)*ones(c,1)'+Ft+ones(nt,1)*b';
    Ft = Ft .* ((Ztz + Zsf + Fd)./(Ztf + Zsz + Fw));
    Ft=real(Ft);
    %=======更新和Ft相关的参数
    Ftt=[ones(nt,1),Ft];
    Ntt=diag([nt,sum(Ft)]);
    Y=[Ys_onehot;Ft];
    
    
    
    %======更新投影矩阵P
    %用b/A替换b*inv(A),用A\b替换inv(A)*b;
    T11=Xs*Yss/Nss;
    T22=Xt*Ftt/Ntt;    
    T1=T11*T11';
    T2=-T11*T22';
    T3=-T22*T11';
    T4=T22*T22';
    %=====T，G矩阵求解
    T=[T1,T2;T3,T4];
    G0=zeros(d,d);
    for t=1:d
        pi=norm(Ps(t,:)-Pt(t,:));
        if pi==0
            G0(t,t)=0;
        else
            G0(t,t)=1/pi;
        end
    end
    G=[[G0,-G0];[-G0,G0]];
        
    
    %======AP+PB=C的形式求P
    A=X*H*X';
    B=T+alpha*G;
    C=W*W';
    D=X*H*Y*W';
    P=sylvester(A\B,C,A\D);
    Ps=P(1:d,:);
    Pt=P(d+1:2*d,:);
    
    %=========目标函数的值。
    %=========用b/A替换b*inv(A),用A\b替换inv(A)*b;
    XA = Ps'*Xs*Yss/Nss-Pt'*Xt*Ftt/Ntt;
    M = Ps - Pt;
    for i=1:d
        pi =[pi;norm(M(i,:))];
    end
    P_21 = norm(pi,1);
    obj(iter,1) = norm(XA,'fro')^2 + alpha * P_21 + norm((X'*P*W+ones(n,1)*b'-Y),'fro')^2+beta*norm(W,'fro')^2;
    [~,predict_label] = max(Ft,[],2);
    acc = length(find(predict_label == Yt))./length(Yt);
    
    if iter==1
        acc_best=acc;
        Ft_final=Ft;
        Ps_final=Ps;
        Pt_final=Pt;
        b_final=b;
        W_final=W;
        obj_final=obj;
        
    elseif iter>1&&acc_best<acc
        acc_best=acc;
        Ft_final=Ft;
        Ps_final=Ps;
        Pt_final=Pt;
        b_final=b;
        W_final=W;
        obj_final=obj;
    end

    
   fprintf('dim=%d,第%d次，alpha=%0.4f,lambda=%0.4f,beta=%0.4f,The  acc=%0.4f\n',options.p,iter,options.alpha,options.lambda,options.beta,acc);
    
end
end

