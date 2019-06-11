raw_dir = 'data/raw_samples';               % Path to raw iris images
circle_dir = 'data/circles/ND-IRIS-0405';   % Path to detected iris circles
norm_dir = 'data/norm';                     % Place to save normalized iris images
feat_dir = 'data/feat';                     % Place to save features;
mask_dir = 'data/mask';                     % Place to save masks;

weights = 'models/UniNet_ND.caffemodel'; % Trained network model
model = 'models/UniNet_deploy.prototxt'; % Network structure prototxt

% Normalize iris images
normalize_iris;

% Extract features and masks from input images
extract_feat;

% Perform all-to-all matching
match_all_to_all;

% Obtaining: 
% g_scores - Matching scores for genuine pairs and 
% i_scores - Matching for imposter pairs
% and other intermediate variables