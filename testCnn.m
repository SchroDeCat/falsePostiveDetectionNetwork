%% transfer learning
%读取训练集和测试集
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
digitData = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
[trainDigitData,testDigitData] = splitEachLabel(digitData,0.5,'randomize');
%显示前20个训练照片
numImages = numel(trainDigitData.Files);
idx = randperm(numImages,20);
for i = 1:20
    subplot(4,5,i)

    I = readimage(trainDigitData, idx(i));

    imshow(I)
end
% 获取matlab自己训练好的网络
load(fullfile(matlabroot,'examples','nnet','LettersClassificationNet.mat'))
% 改变输出层的类别个数
layersTransfer = net.Layers(1:end-3);
% 显示新的类别个数
numClasses =  numel(categories(trainDigitData.Labels));
% 把最后三层替换成新的类别
layers = [...
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];
optionsTransfer = trainingOptions('sgdm',...
    'MaxEpochs',5,...
    'InitialLearnRate',0.0001,...
    'ExecutionEnvironment','cpu');
% 训练网络
netTransfer = trainNetwork(trainDigitData,layers,optionsTransfer);
% 显示测试准确率
YPred = classify(netTransfer,testDigitData);
YTest = testDigitData.Labels;
accuracy = sum(YPred==YTest)/numel(YTest);
% 显示测试结果
idx = 501:500:5000;
figure
for i = 1:numel(idx)
    subplot(3,3,i)

    I = readimage(testDigitData, idx(i));
    label = char(YTest(idx(i)));

    imshow(I)
    title(label)
end