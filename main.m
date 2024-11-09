clear all;
close all;
clc

%% Four domains: { Caltech10, amazon, webcam, dslr }
src = 'Caltech10';
tgt = 'amazon';
nPerClass = 20;

load(['data/' src '_SURF_L10.mat']);     % source domain
fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
Xs = zscore(fts,1);    clear fts
Ys = labels;           clear labels

load(['data/' tgt '_SURF_L10.mat']);     % target domain
fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
Xt = zscore(fts,1);     clear fts
Yt = labels;            clear labels

%%
%methods = {'SVM','TCA','CBSSTL'}; 
methods = {'CBSSTL'};
mu = 5;
lambda =10;
dim = 9;
p1 = 1e-2;
accall=zeros(10,1);
for k = 1:length(methods)
    method = methods{k};
    disp(method)
    for seed=1:10 
    acc=DA(method, mu, lambda, dim, p1, Xs, Ys, Xt, Yt, nPerClass, seed);
    accall(seed,:)=acc;
    end
accave=mean(accall)
accstd=std(accall)
end
