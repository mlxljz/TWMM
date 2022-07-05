function [error] = affine_error_cpt(sz,p_real,p_cpt)
[x,y]    = meshgrid(1:sz(2),1:sz(1));
x_shift  = (1+max(x(:)))/2;
y_shift  = (1+max(y(:)))/2;
x        = x-x_shift;
y        = y-y_shift;
x1       = p_real(1)*x + p_real(2)*y + p_real(3);
y1       = p_real(4)*x + p_real(5)*y + p_real(6);
x2       = p_cpt(1)*x + p_cpt(2)*y + p_cpt(3);
y2       = p_cpt(4)*x + p_cpt(5)*y + p_cpt(6);
error    = sum(sum(sqrt((x2-x1).^2+(y2-y1).^2)))/(sz(1)*sz(2));
end

