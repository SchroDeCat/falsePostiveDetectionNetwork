%% Preprocessing
clear;
imsize = 128;
avg_pooling_size = imsize / 32;
%��ȡѵ�����Ͳ��Լ�
digitDatasetPath = 'E:\��ɽ��ѧ\����\LAB\Breast Cancer\2017-2018����ѧ��\falsePositiveDetection\CancerDetectionImgs\CancerDetectionImgs';
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
% imds.ReadFcn = @(loc)imresize(imread(loc), [224,224,3]); % read while resizing
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

%% define network
% ��ȡmatlab�Լ�ѵ���õ�����
net = googlenet;
% inputSize = net.Layers(1).InputSize;
lgraph = layerGraph(net);
% ��ʾ�µ�������
numClasses =  numel(categories(imds.Labels));
% ����������滻���µ����
lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);
lgraph = connectLayers(lgraph,'pool5-drop_7x7_s1','fc');
% �滻�����
newLayers2 = [
    imageInputLayer([imsize imsize, 1],'Name','data2')
    convolution2dLayer([7,7],64,'Name','conv1-7x7_s3')];
lgraph = removeLayers(lgraph, {'data','conv1-7x7_s2'});
lgraph = addLayers(lgraph,newLayers2);
lgraph = disconnectLayers(lgraph, 'classoutput', 'data2');
lgraph = connectLayers(lgraph,'conv1-7x7_s3','conv1-relu_7x7');
% �滻��Ӧavg pooling
avgPoolingLayer = averagePooling2dLayer([avg_pooling_size avg_pooling_size],'Name','Average Pooling2');
lgraph = removeLayers(lgraph,{'pool5-7x7_s1'});
lgraph = addLayers(lgraph, avgPoolingLayer);
lgraph = connectLayers(lgraph, 'inception_5b-output','Average Pooling2');
lgraph = connectLayers(lgraph, 'Average Pooling2','pool5-drop_7x7_s1');
% figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
% plot(lgraph)
% ylim([0,70])
%����ѡ��
options = trainingOptions('sgdm', ...
    'MiniBatchSize',50, ...
    'MaxEpochs',8,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.2,...
    'LearnRateDropPeriod',5,...
    'InitialLearnRate',1e-4, ...
    'ValidationData',imdsValidation,...
    'ValidationFrequency',30, ...
    'ValidationPatience',10, ...
    'Verbose',true ,...
    'Momentum',0.9,...
    'Plots','training-progress',...
    'ExecutionEnvironment','gpu');  %GPU or not
net = trainNetwork(imdsTrain,lgraph,options);
save('googlenetResult2','net');
%% ��ʾ����׼ȷ��
[YPred,probs] = classify(net,imdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)