% raw_dir = 'data/raw_samples';
% circle_dir = 'data/circles/ND-IRIS-0405';
% norm_dir = 'data/norm';

reso = [64, 512];

if ~isdir(norm_dir); mkdir(norm_dir); end

files = dir([raw_dir, '/*']);

fprintf('Normalizing irises...\n');
for i = 1:length(files)
    [str, type] = strtok(files(i).name, '.');
    if ~isempty(str)
        im = imread([raw_dir, '/', files(i).name]);
        im = im(:, :, 1);
        load([circle_dir, '/', str, '.mat']);
        im_norm = normaliseiris(im, center(1), center(2), radius, ...
        center_p(1), center_p(2), radius_p, reso(1), reso(2));
        imwrite(im_norm, [norm_dir, '/', str, '.bmp']);
        fprintf('%d\n', i);
    end
end