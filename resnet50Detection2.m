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
net = resnet50();
inputSize = net.Layers(1).InputSize;
lgraph = layerGraph(net);
% figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
% plot(lgraph)
% ��ʾ�µ�������
numClasses =  numel(categories(imds.Labels));
% ����������滻���µ����
lgraph = removeLayers(lgraph, {'ClassificationLayer_fc1000', 'fc1000_softmax', 'fc1000',});
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);
lgraph = connectLayers(lgraph,'avg_pool','fc');
% �滻�����
newLayers2 = [
    imageInputLayer([imsize imsize, 1],'Name','data2')
    convolution2dLayer([7,7],64,'Name','conv1-7x7_s3')];
lgraph = removeLayers(lgraph, {'input_1', 'conv1'});
lgraph = addLayers(lgraph,newLayers2);
lgraph = connectLayers(lgraph,'conv1-7x7_s3','bn_conv1');
% �滻��Ӧavg pooling
avgPoolingLayer = averagePooling2dLayer([avg_pooling_size avg_pooling_size],'Name','Average Pooling2');
lgraph = removeLayers(lgraph,{'avg_pool'});
lgraph = addLayers(lgraph, avgPoolingLayer);
lgraph = connectLayers(lgraph, 'activation_49_relu','Average Pooling2');
lgraph = connectLayers(lgraph, 'Average Pooling2','fc');

%����ѡ��
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',8,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.2,...
    'LearnRateDropPeriod',5,...
    'InitialLearnRate',1e-4, ...
    'ValidationData',imdsValidation,...
    'ValidationFrequency',30, ...
    'ValidationPatience',inf, ...
    'Verbose',true ,...
    'Momentum',0.9,...
    'Plots','training-progress',...
    'ExecutionEnvironment','gpu');  %GPU or not
net = trainNetwork(imdsTrain,lgraph,options);
save('resnet50Result425','net')
%% ��ʾ����׼ȷ��
% [YPred,probs] = classify(net,imdsValidation); % error��out of memory 
% accuracy = mean(YPred == imdsValidation.Labels)
testSize = numel(imdsValidation.Files);
count = 0;
for i = 1:testSize
    testImage = imread(char(imdsValidation.Files(i)));
    if classify(net, testImage) == imdsValidation.Labels(i)
        count = count + 1;
    end
end
accuracy = count / testSize