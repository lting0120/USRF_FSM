clear all
clc
warning off;
addpath(genpath('./ClusteringEvaluation'));
addpath(genpath('./SimilarityMatrixConstruction'));
path_data = './Datasets/';
res_path = './Result_temp/';

DataName = {'100Leaves'};
% DataName = {'Yale','MSRC_V1','ORL','100Leaves','NUS','Mnist4'};

datacnt = size(DataName,2);
for name = 1:datacnt
    load([path_data, DataName{name}],'X','truth');
    numclass = length(unique(truth));
    n = size(X,1);
    ker_num = size(X,2);

    result = zeros(11,4); 
    DetailResult = cell(1,length(NNrate));

    runcount = 1;
    NNrate = 0.1:0.1:0.9;
    theta = 2.^(-15:3:15);
    beta = 0.001;

    for i = 1:length(NNrate)   
            [SimMat] = V1_LocalKernelCalculation(X, numclass, NNrate(i));
            [Laplacian, Degree, NorKernel] = laplacian_generation(SimMat);
            for para1 = 1:length(theta)
                for para2 = 1:length(beta)
                    fprintf(1, 'running the proposed algorithm with theta %d..., beta %d...\n', theta(para1),beta(para2));
                    [Ypre, H, obj_main, changed] = USRF_FSM(Laplacian,Degree,NorKernel,numclass,theta(para1),beta(para2));
                    result(para1,:) = clustermatch(Ypre, truth)
                end
            end
            DetailResult{i} = result;      
    end   
    save([res_path,DataName{name},'_result.mat'],'DetailResult');
end