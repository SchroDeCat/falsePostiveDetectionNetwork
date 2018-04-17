clear;
imsize = 100;
%��ȡѵ�����Ͳ��Լ�
digitDatasetPath = '/Users/fengxuezhang/Documents/Lab/CancerDetectionImgs';
digitData = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
[trainDigitData,testDigitData] = splitEachLabel(digitData,0.5,'randomize');

% ͼ��ߴ��һ����RGB2Gray
negativeImgs = dir(fullfile(digitDatasetPath, 'negative'))
positiveImgs = dir(fullfile(digitDatasetPath, 'positive'));
for i = 4:size(negativeImgs)
    imgName = char(strcat(digitDatasetPath,'/', 'negative','/', cellstr(negativeImgs(i).name)));
    img = imread(imgName);
    imgSize = size(img);
    if imgSize(1) < imsize || imgSize(2) < imsize
        delete(imgName);
    else
        img = img(1:imsize, 1:imsize, 1:3);
        
        img = rgb2gray(img);
        imwrite(img,imgName);
    end
end