function [ img,info ] = syfun_linearstrech_nor( img )
info.min = min(min(img));
info.max = max(max(img));
img      = (img-info.min)/((info.max)-info.min);
end

