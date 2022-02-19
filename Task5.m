%% Task 5 Test

clear all; close all;

%% Setup
%load images
Im1 = imread('FD/1.jpg');
Im2 = imread('FD/2.jpg');
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
[indexPairs metric] = matchFeatures(features1,features2,'Metric','SAD','MatchThreshold',10,'Unique',true);
%takes locations of matched KPs
matchedPoints1 = valid_points1(indexPairs(:,1),:);
matchedPoints2 = valid_points2(indexPairs(:,2),:);

x = 0;  %sets while statement to false
while x == 0
    
    %finds fundamental matrix using RANSAC
    [fMatrix, epipolarInliers, status] = estimateFundamentalMatrix(matchedPoints1,matchedPoints2,'Method','RANSAC','NumTrials',2000,'DistanceThreshold',0.001);
    %ensures enoguh correspondences used and epipole is inside image
    if status == 0 && isEpipoleInImage(fMatrix, size(I1)) ~= 1 ...
      && isEpipoleInImage(fMatrix', size(I2)) ~= 1
      x = 1;
    end
end

%gets inlier locations 
inlierPoints1 = matchedPoints1(epipolarInliers, :);
inlierPoints2 = matchedPoints2(epipolarInliers, :);

%finds transformations that align epipolar lines to be parallel
[t1, t2] = estimateUncalibratedRectification(fMatrix, inlierPoints1.Location, inlierPoints2.Location, size(I2));
tform1 = projective2d(t1);
tform2 = projective2d(t2);

%applies transformations to image planes
[I1Rect, I2Rect] = rectifyStereoImages(Im1, Im2, tform1, tform2 , 'FillValues' , 3);

%pick best matches
numBest = 10;
[best indexBest] = mink(metric,numBest);
%gets locations of best KP matches 
PT1 = valid_points1(indexPairs(indexBest,1),:);
PT2 = valid_points2(indexPairs(indexBest,2),:);

%epipole and epipolar lines figures
figure; 
subplot(131);
imshow(I1Rect); 
title('Source Image' ,'FontSize' , 18);
hold on;

lines1 = epipolarLine(fMatrix',PT2.Location);
points1 = lineToBorderPoints(lines1,size(Im1));
line(points1(:,[1,3])',points1(:,[2,4])');
hold off;

subplot(132); 
imshow(I2Rect);
title('Target Image' , 'FontSize',18);
hold on;

lines2 = epipolarLine(fMatrix,PT1.Location);
points2 = lineToBorderPoints(lines2,size(Im2));
line(points2(:,[1,3])',points2(:,[2,4])');
hold off;
truesize;
%also shows rectified result
subplot(133);
imshow(stereoAnaglyph(I1Rect, I2Rect));
title('Rectified Stereo Images' , 'FontSize' , 18);

%% Depth Map

%get the disparity map from rectified images
disparityRange = [0 32];
disparityMap = disparitySGM(rgb2gray(I1Rect),rgb2gray(I2Rect),'DisparityRange',disparityRange,'UniquenessThreshold',0);

%define parameters
focal = 94;
baseline = 4.5;

%thresholds out extreme cases and NaN
disparityMap( disparityMap > 32 ) = 32;
disparityMap( disparityMap < 1 ) = 1;
disparityMap(isnan(disparityMap) ) = 16;

%finds depth map
depthMap = (focal*baseline)./ disparityMap;

%displays disparity and depth maps
figure;
subplot(121);
imshow(disparityMap, [0,32]);
title('Disparity Map','FontSize',18);
colormap jet
colorbar

subplot(122);
imshow(depthMap , [ 0 , 500]);
title('Depth Map','FontSize',18);
colorbar
colormap jet






