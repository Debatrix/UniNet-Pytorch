function masks = load_all_mask_cell( mask_dir, feat_dir, max_num )
%LOAD_ALL_FEAT Summary of this function goes here
%   Detailed explanation goes here

if nargin < 3
    max_num = Inf;
end

files = dir([feat_dir, '/*.mat']);
n = min(max_num, length(files));

masks = cell(n, 1);

for i = 1:n
    str = strtok(files(i).name, '.');
    fp = [mask_dir, '/', str, '.bmp'];
    if exist(fp, 'file')
        im = imread(fp);
        masks{i} = im(:, :, 1) > 0;
    else
        masks{i} = [];
    end
end

end
