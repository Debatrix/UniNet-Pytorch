%norm_dir = 'data/norm';
%feat_dir = 'data/feat';
%mask_dir = 'data/mask';

%weights = 'models/UniNet_ND.caffemodel';
%model = 'models/UniNet_deploy.prototxt';

if ~isdir(feat_dir); mkdir(feat_dir); end;
if ~isdir(mask_dir); mkdir(mask_dir); end;

files = dir([norm_dir, '/*.bmp']);

% Initialize Caffe
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(0);
net = caffe.Net(model, weights, 'test');

fprintf('Extracting features and masks...\n');
for i = 1:length(files);
    im = im2double(imread([norm_dir, '/', files(i).name]));
    im = adjust_iris(im); % Simple enhancement
    im = cat(3, im, im);
    im = permute(im, [2 1 3]); % width <=> height
    
    % Feed image to network & get output
    net.forward({im*255/256}); % Slight scale adjustment due to legacy issue
    output = net.blobs('fuse_a').get_data();
    mask = net.blobs('m_fuse_a').get_data();
    mask = (mask(:, :, 2) > mask(:, :, 1))';
    
    % Save feature and mask
    str = strtok(files(i).name, '.');
    save([feat_dir, '/', str, '.mat'], 'output');
    imwrite(mask, [mask_dir, '/', str, '.bmp']);
    fprintf('%d\n', i);
end