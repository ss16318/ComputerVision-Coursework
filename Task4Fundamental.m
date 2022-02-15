%% Task 4: Fundamental Estimation

clear all; close all;
%% Setup

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
[indexPairs metric] = matchFeatures(features1,features2,'MatchThreshold',0.9,'Unique',true,'MaxRatio',0.2);

%pick all matches
numBest = length(indexPairs);
[best indexBest] = mink(metric,numBest);

%find locations of matches in each image
matchedPoints1 = valid_points1(indexPairs(indexBest,1),:);
matchedPoints2 = valid_points2(indexPairs(indexBest,2),:);

%% Fundamental Matrix 
PT1 = matchedPoints1.Location(:,:);
PT2 = matchedPoints2.Location(:,:);

%code from to calculate fundamental matrix MATLAB using LMedS
[F, inliers, epistatus] = estimateFundamentalMatrix(matchedPoints1,matchedPoints2,'Method','LMedS');

%plot epipole and epipolar lines
figure; 
subplot(121);
imshow(Im1); 
title('Source Image', 'Fontsize',18);
hold on;
plot(PT1(inliers,1),PT2(inliers,2),'go')
lines1s = epipolarLine(F',PT2(inliers,:));
points1s = lineToBorderPoints(lines1s,size(Im1));
line(points1s(:,[1,3])',points1s(:,[2,4])');
hold off;

subplot(122);
imshow(Im2);
title('Target Image', 'Fontsize',18);
hold on;

plot(PT2(inliers,1),PT2(inliers,2),'go')
lines2s = epipolarLine(F,PT1(inliers,:));
points2s = lineToBorderPoints(lines2s,size(Im2));
line(points2s(:,[1,3])',points2s(:,[2,4])');

hold off;
truesize;

