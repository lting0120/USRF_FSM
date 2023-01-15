function [B] = FastmultiCLR(X,c,anchor_rate,opts,rate,isGraph)
% Input:
%       - X: the data matrix of size nSmp * nFea * nView, where each row is a sample
%               point;
%       - c: the number of clusters;
%       - anchor_rate: the rate of sampling data points as anchors;
%       - opts: options for this algorithm
%           - style: 
%               - '1': use centers of clusters generated by kmeans (default);
%               - '2': use randomly sampled points from the original data set;
%               - '3': use the nearest point of each cluster center generated by kmeans;  
%               - '4': use centers of clusters generated by kmeans; 
%           - toy:
%               - '0'; test real data (default);
%               - '1': test toy data;
%           - IterMax: the maximum number of iterations;
% Output:
%       - P: the optimized graph compatible across multiple views;
%       - alpha: the weight assigned for each view;
%       - label: the cluster assignment for each point;
% 
% Please refer to:
%
%	Xuelong Li, Han Zhang, Rong Wang, Feiping Nie, "Multi-view graph clutering: a scalable and parameter-free graph fusion method"
%
%   Written by Han Zhang (zhanghang9937 AT gmail.com), written in 2018/10/15, revised in 2020/6/30
if (~exist('opts','var'))
   opts. style = 1;
   opts. toy = 0;
   opts. IterMax = 50;
end
if nargin < 6
    isGraph = 0;
end 
IterMax = opts.IterMax;
if isGraph == 1
    B = X;
    n_view = length(X);
    [n,m] = size(X{1});
else
%     smp_num = size(X{1},1);
%     per_num = round(0.5*smp_num/c);
    k = rate;
%     k = per_num;
    if isfield(opts,'k')    % 确定是否是结构体数组字段的名称
        k = opts.k;
    end
    n_view = length(X);
    n = size(X{1},1);
    XX = [];
    for v = 1:length(X)
%        for  j = 1:n
%            X{v}(j,:) = ( X{v}(j,:) - mean( X{v}(j,:) ) ) / std( X{v}(j,:) ) ;
%        end
       XX = [XX X{v}];
    end
    m = fix(n*anchor_rate);
%     m = anchor_rate;
    B = cell(n_view,1);
    centers = cell(n_view,1);
%%
%     disp('----------Anchor Selection----------');
%     tic;
    if opts. style == 1 % direct sample
        [~,ind,~] = graphgen_anchor(XX,m);
        for v = 1:n_view
        centers{v} = X{v}(ind, :);
        end
    elseif opts. style == 2 % rand sample
        vec = randperm(n);
        ind = vec(1:m);
        for v = 1:n_view
            centers{v} = X{v}(ind, :);
        end
    elseif opts. style == 3 % KNP
        XX = [];
        for v = 1:n_view
            XX = [XX X{v}];
        end
        [~, ~, ~, ~, dis] = litekmeans(XX, m);
        [~,ind] = min(dis,[],1);
        ind = sort(ind,'ascend');
        for v = 1:n_view
            centers{v} = X{v}(ind, :);
        end
    elseif opts. style == 4 % kmeans sample
        XX = [];
        for v = 1:n_view
           XX = [XX X{v}];
           len(v) = size(X{v},2);
        end
        [~, Cen, ~, ~, ~] = litekmeans(XX, m);
        t1 = 1;
        for v=1:n_view
           t2 = t1+len(v)-1;
           centers{v} = Cen(:,t1:t2);
           t1 = t2+1;
        end
    end
%     toc;
%     tic;
%%
%     disp('----------Single Graphs Inilization----------');
    for v = 1:n_view
        D = L2_distance_1(X{v}', centers{v}');
        [~, idx] = sort(D, 2); % sort each row
        B{v} = zeros(n,m);
        for ii = 1:n
            id = idx(ii,1:k+1);
            di = D(ii, id);
            B{v}(ii,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
        end
    end
%     toc;
end

end
