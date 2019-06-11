function im_ad = adjust_iris(im, r_th)

if nargin < 2
    r_th = [0.05, 0.95];
end

% reflection removal
if ~isa(im, 'double'); im = im2double(im); end

th = stretchlim(im, [0 0.99]);
rfl = im >= th(2);
rfl = bwareaopen(rfl, round(0.005*numel(im)));
rfl = imdilate(rfl, strel('disk', 7));

th = stretchlim(im(~rfl), r_th);

im_ad = imadjust(im, th);

end