function im_cropped = func_imCrop(im,imSize)

[height,width,~] = size(im);

cy = (1+height)/2;
cx = (1+width)/2;
rh = imSize(1)/2;
rw = imSize(2)/2;

sty = round(cy-rh);
edy = sty + imSize(1) - 1;
stx = round(cx-rw);
edx = stx + imSize(2) - 1;

im_cropped = im(sty:edy,stx:edx,:);