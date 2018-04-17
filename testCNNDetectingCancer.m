%% Preprocessing
clear;
imsize = 100;
%读取训练集和测试集
digitDatasetPath = '/Users/fengxuezhang/Documents/Lab/CancerDetectionImgs';
digitData = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
[trainDigitData,testDigitData] = splitEachLabel(digitData,0.5,'randomize');

%% define network
%网络层次设置
layers = [ ...
    imageInputLayer([imsize imsize 1])
    convolution2dLayer(5,5)
    reluLayer
    crossChannelNormalizationLayer(5,'Alpha',0.00005,'Beta',0.75,'K',1)  %Norm layer1        
    convolution2dLayer(3,3,'Stride',1,'BiasLearnRateFactor',2)         %Cov2 layer
    reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
%网络选项
options = trainingOptions('sgdm', ...
    'MaxEpochs',200, ...
    'InitialLearnRate',1e-4, ...
    'MiniBatchSize',256,...
    'ExecutionEnvironment','cpu');  % 禁用GPU
%定义网络
net = trainNetwork(trainDigitData,layers,options);