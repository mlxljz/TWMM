function [res] = syfun_rgb2gray(img)
if size(img,3)==3
    res = rgb2gray(img);
else
    res = img;
end
end

