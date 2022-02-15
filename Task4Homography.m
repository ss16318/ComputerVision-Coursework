%% Task 4: Homography Estimation

clear all; close all;
%% Setup

%load images
Im1 = imread('HG/2.jpg');
Im2 = imread('HG/5.jpg');
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

%pick best matches
numBest = 10;
[best indexBest] = mink(metric,numBest);

%find locations of matches in each image
matchedPoints1 = valid_points1(indexPairs(indexBest,1),:);
matchedPoints2 = valid_points2(indexPairs(indexBest,2),:);

figure; 
ax = axes;
showMatchedFeatures(Im1,Im2,matchedPoints1,matchedPoints2,'montage','Parent',ax);
title(ax, 'Keypoint correspondence of image pair from HG Set' , 'Fontsize' ,32);
legend(ax, 'Matched Keypoints in Source','Matched Keypoints in Target','Fontsize' , 16);
plotOptions({'ro','g+','y-'});

%% Homogrpahy Matrix 

%finds homogrpahy matrix using RANSAC
[tform inliers1 inliers2 ] = estimateGeometricTransform(matchedPoints1,matchedPoints2,'projective','Confidence',99.99);

%converts homography matrix to correct form
H = tform.T;

%gets inliers and makes points 3D
pts1 = inliers1.Location;
pts2 = inliers2.Location;
z_pt = ones(length(pts2(:,1)),1);

pts1 = [pts1 z_pt];
pts2 = [pts2 z_pt];

% applies approriate transpositions
pts1_T = pts1.';
H_T = H.';

projection = zeros(length(z_pt),3);

%finds eacj KP error
for i = 1:length(z_pt)
    projection(i,:) = H_T * pts1_T(:,i);
end

%finds MSE of all KPs
MSE = immse(double(pts2) , projection )