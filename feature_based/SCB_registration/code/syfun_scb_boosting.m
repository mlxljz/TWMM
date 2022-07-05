function [img]= syfun_scb_boosting(img)
img                 = syfun_linearstrech_nor(img);
img                 = imgaussfilt(img,1);
img                 = (weiner_enhence_GGD_org(img,3,0.02345,5.54522,0.41972));
end