%% Preprocessing
clear;
imsize = 80;
%��ȡѵ�����Ͳ��Լ�
digitDatasetPath = 'E:\��ɽ��ѧ\����\LAB\Breast Cancer\2017-2018����ѧ��\falsePositiveDetection\CancerDetectionImgs\CancerDetectionImgs';
digitData = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
[trainDigitData,testDigitData] = splitEachLabel(digitData,0.8,'randomize');

%% define network
% % ��ȡmatlab�Լ�ѵ���õ�����
% net = resnet50();
% % �ı�������������
% layersTransfer = net.Layers(1:end-3);
% % ��ʾ�µ�������
% numClasses =  numel(categories(trainDigitData.Labels));
% % ����������滻���µ����
% layers = [...
%     layersTransfer
%     fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
%     softmaxLayer
%     classificationLayer];
% %����������
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
%����ѡ��
options = trainingOptions('sgdm', ...
    'MaxEpochs',150, ...
    'InitialLearnRate',1e-3, ...
    'MiniBatchSize',128,...
    'ExecutionEnvironment','gpu');  % ����GPU
%��������
net = trainNetwork(trainDigitData,layers,options);

%% ��ʾ����׼ȷ��
YPred = classify(net,testDigitData);
YTest = testDigitData.Labels;
accuracy = sum(YPred==YTest)/numel(YTest)