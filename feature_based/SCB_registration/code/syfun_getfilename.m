function [ filename ] = syfun_getfilename( pathname,cut)
tmp      = dir(pathname);
filename = {tmp.name};
if cut
    filename(1:2) = [];
end
end

