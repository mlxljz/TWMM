function [p]=SCB_demo_python(img1,img2)

addpath(genpath('code'));

img1 = im2double(img1);
img2 = im2double(img2);
img1_scb  = syfun_scb_boosting(img1);
img2_scb  = syfun_scb_boosting(img2);
p     = ssd_affine(img1_scb,img2_scb);
end