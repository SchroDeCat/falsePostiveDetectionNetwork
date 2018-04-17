%% transfer learning
%��ȡѵ�����Ͳ��Լ�
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
digitData = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
[trainDigitData,testDigitData] = splitEachLabel(digitData,0.5,'randomize');
%��ʾǰ20��ѵ����Ƭ
numImages = numel(trainDigitData.Files);
idx = randperm(numImages,20);
for i = 1:20
    subplot(4,5,i)

    I = readimage(trainDigitData, idx(i));

    imshow(I)
end
% ��ȡmatlab�Լ�ѵ���õ�����
load(fullfile(matlabroot,'examples','nnet','LettersClassificationNet.mat'))
% �ı�������������
layersTransfer = net.Layers(1:end-3);
% ��ʾ�µ�������
numClasses =  numel(categories(trainDigitData.Labels));
% ����������滻���µ����
layers = [...
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];
optionsTransfer = trainingOptions('sgdm',...
    'MaxEpochs',5,...
    'InitialLearnRate',0.0001,...
    'ExecutionEnvironment','cpu');
% ѵ������
netTransfer = trainNetwork(trainDigitData,layers,optionsTransfer);
% ��ʾ����׼ȷ��
YPred = classify(netTransfer,testDigitData);
YTest = testDigitData.Labels;
accuracy = sum(YPred==YTest)/numel(YTest);
% ��ʾ���Խ��
idx = 501:500:5000;
figure
for i = 1:numel(idx)
    subplot(3,3,i)

    I = readimage(testDigitData, idx(i));
    label = char(YTest(idx(i)));

    imshow(I)
    title(label)
end