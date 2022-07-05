clc,clear,close all;
addpath(genpath('code'));
%% 
p_simulate      = [1.15 0.15 -15,-0.15 1.15 15]';
path            = 'image_data/rgb_nir/';
imgall          = syfun_getfilename(path,1);
%%
img1 = syfun_rgb2gray(imread(fullfile(path,imgall{1})));
img2 = syfun_rgb2gray(imread(fullfile(path,imgall{2})));
sz   = size(img1);
img1 = (imresize(im2double(img1),300/min(sz(1),sz(2))));
img2 = (imresize(im2double(img2),300/min(sz(1),sz(2))));
%%
img2 = affine_transform(img2,p_simulate);
%% direct registration
img1_org  = func_imCrop(img1,[256 256]);
img2_org  = func_imCrop(img2,[256 256]);
p         = ssd_affine(img1_org,img2_org);
error_org = affine_error_cpt(size(img4),inverse_affine_param(p_simulate),p)
figure,imshowpair(img1,affine_transform(img2,p));
title('ORG img registration result');
%% SCB registration
img1_scb  = syfun_scb_boosting(img1);
img2_scb  = syfun_scb_boosting(img2);
img1_scb  = func_imCrop(img1_scb,[256 256]);
img2_scb  = func_imCrop(img2_scb,[256 256]);
p     = ssd_affine(img1_scb,img2_scb);
error_scb = affine_error_cpt(size(img1),inverse_affine_param(p_simulate),p)
figure,imshowpair(img1,affine_transform(img2,p));
title('SCB img registration result');