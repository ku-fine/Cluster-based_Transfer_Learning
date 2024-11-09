%ITCA is the former name of IGLDA
function [accuracy]= DA(method, mu, lambda, dim, p1, Xs, Ys, Xt, Yt, nPerClass,seed)
%The four different methods: 'SVM','TCA','SSTCA','ITCA'
%   'SVM':Support vector machine, if you choose this, and the algorithm will take the SVM as classifier;
%   'TCA':transfer component analysis
%   'SSTCA':semisupervised TCA, and some parameters should be set in the getWSS.m
%   'ITCA':IGLDA
%ind is the number of the target data
%mu lambda are balance factors
%dim dimension of the eigenvalue
%p1 the kernel function.

  
inds = split(Ys, nPerClass);
Xs = Xs(inds,:);
Ys = Ys(inds);

Xs = Xs';
Ys = Ys';
Xt = Xt';
Yt = Yt';
    
%%%%%%%%%%% 使用不同方法处理数据 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~strcmp(method,'SVM')%用不同的TCA方法将数据映射到新的空间
    
%%%%% 设置不同的核函数(将选择的方法解除注释，未选择的方法注释掉)
    %选用Laplacian核函数; use Laplacian kernel function
 %    p2=0;p3=0;
%     kind='Laplacian';

    %选用高斯核函数; use Gaussian kernel function
   %  p2=0;p3=0;
   %  kind='Gaussian';

    %选用多项式核函数; use Polynomial kernel function 默认用的是这个
   p2=0;p3=1;
   kind='Polynomial';
    

%%%%% 获取W; the following code is to get the latent space - the martix W can help us 
%%%%% to map data to the space. This is key part of IGLDA
    switch method
        case 'TCA'
            [W, K, ns, nt]=getW(Xs, Xt, mu, lambda, dim, kind, p1, p2, p3);
        case 'SSTCA'
            [W, K, ns, nt]=getWSS(Xs, Xt, mu, lambda, dim, kind, p1, p2, p3, Ys);
        case 'CBSSTL'
            [W, K, ns, nt]=getWCBSSTL(Xs, Xt, mu, lambda, dim, kind, p1, p2, p3, Ys, Yt);
    end
    
%%%%% 将数据映射到新空间; Map the data into the latent space by using W
    [Xs_, Xt_]=getY(W, K, ns, nt);
    Xt1=getNewY(Xs, Xt, Xt, W, kind, p1, p2, p3);%获取了新的Xt，虽然和之前的一样，但是为了假如有新的数据
    
else%直接使用SVM; Or to use SVM directly
    %X=allsamples;
    Xt1=Xt;
    Xs_=Xs;
end



%%%%%%%%%%% to classify the mapped data by using SVM directly%%%%%%%%%%%%%%%%
    rng(seed)%种子
    param = '-t 0 -q';
    if strcmp(method,'CBSSTL')
        newYt=getNewYt(Xs_, Xt_,Ys,Yt);
        model = svmtrain(newYt, Xt_', param);
    else
        model = svmtrain(Ys', Xs_', param);
    end
     [~, accuracy, ~] = svmpredict(Yt', Xt1', model);
     accuracy = accuracy(1);
end

%% Try to let every  category of the data has nPerClass samples 

function idx = split(Y,nPerClass)
idx = [];
for C = 1 : max(Y)
    idxC = find(Y == C);
    rn = randperm(length(idxC));
    idx = [idx; idxC( rn(1:min(nPerClass,length(idxC))))];
end
end

