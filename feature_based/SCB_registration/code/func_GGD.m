function [GGD] = func_GGD(x,mu,alpha,beta,c)
GGD = c*exp(-abs((x - mu)/beta).^alpha);
end

