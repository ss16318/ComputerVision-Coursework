%% Task 2: Finding KP correspondence

clear all; close all;

%% Automatic matching 

%load images
Im1 = imread('/Users/KevinWang/Desktop/CVPR/FD/2.jpg');
Im2 = imread('/Users/KevinWang/Desktop/CVPR/FD/10.jpg');

%converts images from RGB to grayscale
I1 = rgb2gray(Im1);
I2 = rgb2gray(Im2);

%detect features
points1 = detectSURFFeatures(I1);
points2 = detectSURFFeatures(I2);

%get descriptors from each KP
[features1,valid_points1] = extractFeatures(I1,points1);
[features2,valid_points2] = extractFeatures(I2,points2);

%match KPs
[indexPairs metric] = matchFeatures(features1,features2,'MatchThreshold',1,'Unique',true,'MaxRatio',0.2);

%pick best matches
numBest = 10;
[best indexBest] = mink(metric,numBest);

%find locations of matches in each image
matchedPoints1 = valid_points1(indexPairs(indexBest,1),:);
matchedPoints2 = valid_points2(indexPairs(indexBest,2),:);

%display automatic KP correspondance matches
figure; 
ax = axes;
showMatchedFeatures(Im1,Im2,matchedPoints1,matchedPoints2,'montage','Parent',ax);
title(ax, 'Candidate point matches');
legend(ax, 'Matched points 1','Matched points 2');

%% Manual Matching

figure;
subplot(1,2,1)
imshow(Im1)
subplot(1,2,2)
imshow(Im2)

for i = 1:100   %presuming humans cannot match more than 100 pairs of correspondances in 2min
    subplot(1,2,1)
    [x1(i),y1(i)] = ginput(1);
    hold all;
    scatter(x1(i),y1(i),210,'.')    %displaying dots on places chosen manually
    
    subplot(1,2,2)
    [x2(i),y2(i)] = ginput(1);
    hold all;
    scatter(x2(i),y2(i),210,'.')
end