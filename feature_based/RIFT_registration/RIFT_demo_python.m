
function [good]=RIFT_demo_python(img1,img2)

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

disp('nearest matching')
% nearest matching
[indexPairs,matchmetric] = matchFeatures(des_m1.des,des_m2.des,'MaxRatio',1,'MatchThreshold', 100);
matchedPoints1 = des_m1.kps(indexPairs(:, 1), :);
matchedPoints2 = des_m2.kps(indexPairs(:, 2), :);
[matchedPoints2,IA]=unique(matchedPoints2,'rows');
matchedPoints1=matchedPoints1(IA,:);

disp('outlier removal')
%outlier removal
H=FSC(matchedPoints1,matchedPoints2,'affine',2);
Y_=H*[matchedPoints1';ones(1,size(matchedPoints1,1))];
Y_(1,:)=Y_(1,:)./Y_(3,:);
Y_(2,:)=Y_(2,:)./Y_(3,:);
E=sqrt(sum((Y_(1:2,:)-matchedPoints2').^2));
inliersIndex=E<3;
cleanedPoints1 = matchedPoints1(inliersIndex, :);
cleanedPoints2 = matchedPoints2(inliersIndex, :);
good=[cleanedPoints1,cleanedPoints2]

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
