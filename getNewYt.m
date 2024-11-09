function newYt=getNewYt(Xs_,Xt_ ,Ys, Yt)


% labeledClusters 是含标签聚类的集合，每个聚类是一个 NxM 矩阵
% labeledClusterLabels 存储每个含标签聚类的标签
% unlabeledClusters 是不含标签聚类的集合
data=Xs_';
labels =Ys';
pseudoLabels=Yt';
pseudoData=Xt_';
% 假设 data 是一个 m x n 矩阵，其中 m 是数据点的数量，n 是特征的数量
% labels 是一个 m x 1 的矩阵，包含每个数据点的聚类标签

% 获取唯一的聚类标签
uniqueLabels = unique(labels);

% 初始化用于存储每个聚类平均值的矩阵
clusterMeans = zeros(length(uniqueLabels), size(data, 2));

% 计算每个聚类的平均值
for i = 1:length(uniqueLabels)
    % 提取当前聚类的数据点
    currentClusterData = data(labels == uniqueLabels(i), :);
    
    % 计算当前聚类的平均值
    clusterMeans(i, :) = mean(currentClusterData, 1);
end



% 假设 pseudoData 是一个包含伪聚类数据点的矩阵
% pseudoLabels 是一个包含伪聚类标签的 m x 1 矩阵

% 计算伪聚类的平均值
uniquePseudoLabels = unique(pseudoLabels);
pseudoClusterMeans = zeros(length(uniquePseudoLabels), size(pseudoData, 2));

for i = 1:length(uniquePseudoLabels)
    pseudoClusterData = pseudoData(pseudoLabels == uniquePseudoLabels(i), :);
    pseudoClusterMeans(i, :) = mean(pseudoClusterData, 1);
end

% 初始化用于存储最接近的有标签聚类标签的向量
nearestClusterLabels = zeros(size(pseudoData, 1), 1);

% 为每个伪聚类找出最接近的有标签聚类
for i = 1:length(uniquePseudoLabels)
    % 计算当前伪聚类与所有有标签聚类的距离
    distances = sum((clusterMeans - pseudoClusterMeans(i, :)).^2, 2);
    
    % 找出最小距离的有标签聚类
    [~, nearestClusterIndex] = min(distances);
    
    % 将最接近的有标签聚类的标签赋予所有属于当前伪聚类的数据点
    nearestClusterLabels(pseudoLabels == uniquePseudoLabels(i)) = uniqueLabels(nearestClusterIndex);
end

% nearestClusterLabels 现在包含了每个伪聚类最接近的有标签聚类的标签


% clusterMeans 矩阵现在包含每个聚类的平均值



newYt=nearestClusterLabels;
% assignedLabels 包含了不含标签聚类的赋予的标签
