
function [good]=RIFT_points_demo_python(img1,img2)

im1 = img1;
im2 = img2;

if size(im1,3)==1
    temp=im1;
    im1(:,:,1)=temp;
    im1(:,:,2)=temp;
    im1(:,:,3)=temp;
end

if size(im2,3)==1
    temp=im2;
    im2(:,:,1)=temp;
    im2(:,:,2)=temp;
    im2(:,:,3)=temp;
end

disp('RIFT feature detection and description')
% RIFT feature detection and description
[des_m1,des_m2] = RIFT_no_rotation_invariance(im1,im2,4,6,96);
matchedPoints1 = des_m1.kps(:, :);
matchedPoints2 = des_m2.kps(:, :);




good=[matchedPoints1]

%im_Ref=im1;
%min_Ref=double(min(min(min(im_Ref))));
%max_Ref=double(max(max(max(im_Ref))));
%range_Ref=double(max_Ref-min_Ref);
%im1=double(double(im_Ref)-min_Ref)/range_Ref;
%im1=uint8(im1*255);
%figure; showMatchedFeatures(im1, im2, cleanedPoints1, cleanedPoints2, 'montage')
%disp('registration result')
%% registration
%image_fusion(im2,im1,double(H));

end
