function [res] = weiner_enhence_GGD_org(img,varargin)
%% local edge preserving transform
tilesz = varargin{1};
c      = varargin{2};
alpha  = varargin{3};
beta   = varargin{4};

sz       = size(img);
imgmean  = imfilter(img,ones(tilesz,tilesz)/tilesz^2,'same','replicate');

padsz    = floor(tilesz/2);
imgpad   = padarray(img,[padsz,padsz],'both','replicate');
res      = zeros(sz);
%%
for m=1:tilesz
    for n=1:tilesz
        if  m==ceil(tilesz/2)&&n==ceil(tilesz/2)
            continue;
        end
        imgnow = imgpad(m:m+sz(1)-1,n:n+sz(2)-1,:);
        w_mask = (((img - imgnow).^2)./((img - imgnow).^2 + (func_GGD(imgmean,0.5,alpha,beta,c)).^2+1e-50));
        if any(any(isnan(w_mask)))
            keyboard;
        end
        res    = res + w_mask;
    end
end
res = res/(tilesz^2-1);
end

