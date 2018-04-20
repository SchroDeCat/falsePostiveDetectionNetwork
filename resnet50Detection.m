%% Preprocessing
clear;
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
% figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
% plot(lgraph)
ylim([0,10])
%����ѡ��
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',false ,...
    'Plots','training-progress',...
    'ExecutionEnvironment','gpu');  %GPU or not
net = trainNetwork(imdsTrain,lgraph,options);
save('resnet50Result','net')
%% ��ʾ����׼ȷ��
% [YPred,probs] = classify(net,imdsValidation);
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