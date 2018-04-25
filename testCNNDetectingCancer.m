%% Preprocessing
clear;
imsize = 80;
%读取训练集和测试集
digitDatasetPath = 'E:\中山大学\大三\LAB\Breast Cancer\2017-2018春季学期\falsePositiveDetection\CancerDetectionImgs\CancerDetectionImgs';
digitData = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
[trainDigitData,testDigitData] = splitEachLabel(digitData,0.8,'randomize');

%% define network
% % 获取matlab自己训练好的网络
% net = resnet50();
% % 改变输出层的类别个数
% layersTransfer = net.Layers(1:end-3);
% % 显示新的类别个数
% numClasses =  numel(categories(trainDigitData.Labels));
% % 把最后三层替换成新的类别
% layers = [...
%     layersTransfer
%     fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
%     softmaxLayer
%     classificationLayer];
% %网络层次设置
layers = [ ...
    imageInputLayer([imsize imsize 3])
    convolution2dLayer(10,10)
    reluLayer
    crossChannelNormalizationLayer(5,'Alpha',0.00005,'Beta',0.75,'K',1)  %Norm layer1        
%     convolution2dLayer(5,5,'Stride',1,'BiasLearnRateFactor',2)         %Cov2 layer
%     reluLayer
%     convolution2dLayer(3,3,'Stride',1,'BiasLearnRateFactor',2)         %Cov2 layer
%     reluLayer
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
%网络选项
options = trainingOptions('sgdm', ...
    'MaxEpochs',150, ...
    'InitialLearnRate',1e-3, ...
    'MiniBatchSize',128,...
    'ExecutionEnvironment','gpu');  % 禁用GPU
%定义网络
net = trainNetwork(trainDigitData,layers,options);

%% 显示测试准确率
YPred = classify(net,testDigitData);
YTest = testDigitData.Labels;
accuracy = sum(YPred==YTest)/numel(YTest)