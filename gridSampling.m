% moving the sampling area by grid to try to get eoungh sample of positive area and false area
clc,clear;
totalCount = 0;
sampleAmount = 5;
sampleSize = [80, 80];
sampleAreaSize = sampleSize(1) * sampleSize(2) * 255;
halfHeight = sampleSize(1)/2;
halfLength = sampleSize(2)/2;
% folderName = 'F:\Dropbox\USSeg2013\US2DImgs\';
% ---------define positions----------
folderName = '/Users/fengxuezhang/Documents/Lab/US2DImgs/';  % folders of the origin BUSimages
negativeFolderName = '/Users/fengxuezhang/Documents/Lab/CancerDetectionImgs/negative2/'; % folder to store the negative region
positiveFolderName = '/Users/fengxuezhang/Documents/Lab/CancerDetectionImgs/positive2/'; % folder to store the positive region

% ----------create folders-----------
mkdir(negativeFolderName);
mkdir(positiveFolderName);
% ----------init parameters----------
caseStartNum = 1;
caseEndNum = 115;
% ------------processing-------------
for i = caseStartNum:caseEndNum
    sprintf('case %d',i)
    %%% directing to origin BUSimages as /*b_2Value.jpg & /*a.jpg
    goldenStdName = sprintf('%s%d%s',folderName,i,'/*b_2Value.jpg');
    goldenStdBUSFiles = dir(goldenStdName);
    for j = 1:size(goldenStdBUSFiles)
        goldenStdImgName = goldenStdBUSFiles(j).name;
        imgNum = goldenStdImgName(1);
        % sampling
        goldenStdImageName = sprintf('%s%d%c%s',folderName,i,'/',goldenStdImgName);
        originImageName = sprintf('%s%d%c%s',folderName,i,'/',imgNum,'a.jpg');
        if exist(goldenStdImageName,'file') == 0 
            continue;
        end
        std = imread(goldenStdImageName);
        org = imread(originImageName);

        
        % random sampling
        sampleCount = 0;
        sampleNegativeCount = 0;
        unEqualSizeCount = 0;
        imageSize = size(std);
        imageSizeOrg = size(org);
        % transfer rgb to gray
        if size(imageSize) == 3
            std = rgb2gray(std);
        end
        if size(imageSizeOrg) == 3
            org = rgb2gray(org);
        end
        % ignore different Size
        if imageSize(1:2) ~= imageSizeOrg(1:2)
            break;
        end
        CenterX = halfHeight + 1;
        while CenterX < imageSize(2) - halfLength
            CenterY = halfLength + 1;
            while CenterY < imageSize(1) - halfHeight
                sampledAreaGld = std(CenterY - halfHeight:CenterY + halfHeight - 1, ...
                    CenterX - halfLength: CenterX + halfLength - 1);
                sampledAreaOrg = org(CenterY - halfHeight:CenterY + halfHeight - 1, ...
                    CenterX - halfLength: CenterX + halfLength - 1);
                areaSum = sum(sum(sampledAreaGld));
    %             sampleAreaSize * 0.1
    %             imshow(sampledAreaGld)
    %             pause()
                %judge and save
                if sampleNegativeCount <= 2 * sampleAmount && areaSum <= sampleAreaSize * 0.1
                    svSampleImgName = sprintf('%s%d%s',negativeFolderName,totalCount,'.jpg');
                    sampleNegativeCount = sampleNegativeCount + 1;
                elseif areaSum >= sampleAreaSize * 0.5
                    svSampleImgName = sprintf('%s%d%s',positiveFolderName,totalCount,'.jpg');
                    sampleCount = sampleCount + 1;
                else
                    CenterY = CenterY + halfHeight
                    continue;
                end
                totalCount = totalCount + 1;
                imwrite(sampledAreaOrg, svSampleImgName);
                CenterY = CenterY + halfHeight;
            end
            CenterX = CenterX + halfLength;
        end
    end
end


