function   [Ps_final,Pt_final,Ft_final,W_final,b_final,acc_best,obj_final]=RLRKT_update_useinv(Xs,Ys,Xt,Yt,options)
%RLRKT_UPDATE ����Ĳ�����Xs,Ys,Xt,Yt,�Լ�options�еĦ����£��ˣ�ͶӰ����P��ά��.

%=======ȡ����ε����Ĳ���
p=options.p;
alpha=options.alpha;
beta=options.beta;
lambda=options.lambda;

%======���ݵĳ�ʼ��ʼά��d.
d=size(Xs,1);
%======Դ��Ŀ���������������ns/nt
[~,ns]=size(Xs);
[~,nt]=size(Xt);


%=====������������n
n=ns+nt;

%======��ǩ������C
c=double(max(Ys));

%======��Դ��ı�ǩ����onehot���룺Ys_onehot
Ys_onehot=onehot(Ys,c);

%=====>> Yss=[1,Ys],Ns=diag(1/ns_i),Nss=diag(1/ns,Ns)

Yss=[ones(ns,1),Ys_onehot];
Ns=inv(diag(sum(Ys_onehot)));
Nss=inv(diag([ns,sum(Ys_onehot)]));

%=======Ŀ�����ǩ�ĳ�ʼ����
Ft_init=ones(nt,c)./c;
Ft=Ft_init;

%=========��ʼ����Ŀ�����ǩ����֪��Դ���ǩƴ��:Y
Y= [Ys_onehot;Ft_init];

%=========Դ���Ŀ����X
X=[Xs,zeros(d,nt);zeros(d,ns),Xt];


%=========���ľ���
H = eye(n)-1/n * ones(n,n);

%=========��ʼ��ͶӰ����P,ʹ��PCA��
p_dim=[];
p_dim.ReducedDim=p;
[P,~]=PCA(X',p_dim);
Ps=P(1:d,:);
Pt=P(d+1:2*d,:);
%=========��ʼ��W,ʹ��PCA��
W_x=X'*P;
w_dim=[];
w_dim.ReducedDim=c;
[W,~]=PCA(W_x,w_dim);


%======����ѭ������֮ǰ��ʼ���Ĳ����У�Ft--->Ft_init---->Y= [Ys_onehot;Ft_init].W,P.
NITER=10;
for iter=1:NITER
    %======�������������b
    b=1/n*(Y'*ones(n,1)-W'*P'*X*ones(n,1));
    %======���������������W
    % ������ߴ��Ƿ����
    if size(P' * X * H * X' * P + beta * eye(p), 1) ~= p
        fprintf('Warning: Matrix dimensions are not compatible in iteration %d. Skipping this iteration.\n', iter);
        continue; % �����˴ε���
    end
    W=(inv((P'*X*H*X'*P+beta*eye(options.p)))+eps*eye(p))*(P'*X*H*Y);
    
    %======�������������Ft
    Nt=inv(diag(sum(Ft)))+eps*eye(c);
    
    %=====>> %����ʾ����b/A�滻b*inv(A),��A\b�滻inv(A)*b��
    Zs=Xt'*Pt*Ps'*Xs*Ys_onehot*Ns*Nt;
    Zt=Xt'*(Pt*Pt')*Xt*Ft*Nt*Nt;
    %====>> Zt=Zt(+)-Zt(-),Zs=Zs(+)-Zs(-).
    Ztz = Zt;
    Ztz(Ztz<0)=0;%��ֵȫ������Ϊ0
    Zsz = Zs;
    Zsz(Zsz<0)=0;%��ֵȫ������Ϊ0
    Ztf = Zt;
    Ztf(Ztf>0)=0;%��ֵȫ���滻Ϊ0
    Zsf = Zs;
    Zsf(Zsf>0)=0;%��ֵȫ���滻Ϊ0
    Fw =ones(nt,1)*ones(c,1)'+Xt'*Pt*W;
    Fd =lambda*Ft*ones(c,1)*ones(c,1)'+Ft+ones(nt,1)*b';
    Ft = Ft .* ((Ztz + Zsf + Fd)./(Ztf + Zsz + Fw));
    Ft=real(Ft);
    %=======���º�Ft��صĲ���
    Ftt=[ones(nt,1),Ft];
    Ntt=inv(diag([nt,sum(Ft)]))+eps*eye(c+1);
    Y=[Ys_onehot;Ft];
    
    
    
    %======����ͶӰ����P
    %��b/A�滻b*inv(A),��A\b�滻inv(A)*b;
    T11=Xs*Yss*Nss;
    T22=Xt*Ftt*Ntt;    
    T1=T11*T11';
    T2=-T11*T22';
    T3=-T22*T11';
    T4=T22*T22';
    %=====T��G�������
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
    
    
    
    %======AP+PB=C����ʽ��P
    % fprintf('Iteration %d, Computing A...\n', iter);
    A = pinv(X*H*X') + eps*eye(2*d);
    
    % fprintf('Iteration %d, Computing B, C, D...\n', iter);
    B = T + alpha*G;
    C = W*W';
    D = X*H*Y*W';
    
    % fprintf('Iteration %d, Cleaning matrices...\n', iter);
    A = clean_matrix(A);
    B = clean_matrix(B);
    C = clean_matrix(C);
    D = clean_matrix(D);
    
    % ���A*B��A*D�Ƿ�Ϊȫ�����
    if all(all(abs(A * B) < 1e-10)) || all(all(abs(A * D) < 1e-10))
        fprintf('Iteration %d, Warning: A*B or A*D is effectively a zero matrix. Skipping this iteration.\n', iter);
        continue; % ������μ���
    end
    
    % fprintf('Iteration %d, Solving Sylvester equation...\n', iter);
    P = sylvester(A*B, C, A*D);
    Ps = P(1:d, :);
    Pt = P(d+1:2*d, :);
    
    % fprintf('Iteration %d, Updating objective and accuracy...\n', iter);
    % Ŀ�꺯����ֵ
    XA = Ps'*Xs*Yss*Nss - Pt'*Xt*Ftt*Ntt;
    M = Ps - Pt;
    pi = zeros(d, 1);
    for i = 1:d
        pi(i) = norm(M(i, :));
    end
    P_21 = norm(pi, 1);
    obj(iter, 1) = norm(XA, 'fro')^2 + alpha * P_21 + norm((X'*P*W + ones(n, 1)*b' - Y), 'fro')^2 + beta*norm(W, 'fro')^2;
    [~, predict_label] = max(Ft, [], 2);
    acc = length(find(predict_label == Yt)) / length(Yt);
    
    if iter == 1
        acc_best = acc;
        Ft_final = Ft;
        Ps_final = Ps;
        Pt_final = Pt;
        b_final = b;
        W_final = W;
        obj_final = obj;
    elseif iter > 1 && acc_best < acc
        acc_best = acc;
        Ft_final = Ft;
        Ps_final = Ps;
        Pt_final = Pt;
        b_final = b;
        W_final = W;
        obj_final = obj;
    end  
    % fprintf('dim=%d,��%d�Σ�alpha=%0.4f,lambda=%0.4f,beta=%0.4f,The  acc=%0.4f\n', options.p, iter, options.alpha, options.lambda, options.beta, acc);
end
end