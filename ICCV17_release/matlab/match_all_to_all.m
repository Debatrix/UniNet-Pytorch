% feat_dir = 'data/feat';
% mask_dir = 'data/mask';

% Load all features once into memory to save matching time
% If memory is an issue, should load features on-the-fly
[feats, labels] = load_all_feat_cell(feat_dir);
masks = load_all_mask_cell(mask_dir, feat_dir);
n = size(feats, 1);

% Binarize features
for i = 1:n
    masks{i} = masks{i}';
    th = mean(feats{i}(masks{i}));
    masks{i}(abs(feats{i}-th) < 0.6) = false;
    feats{i} = feats{i} > th; 
end

s_mat = zeros(n, n);
r_mat = zeros(n, n); % mask_ratio
idx_mat = zeros(n);

fprintf('Matching...\n');
for i = 1:n
    if isempty(masks{i})
        continue;
    end
    fv1 = feats{i};
    fv1s = shift_map(fv1, 16);
    m1s = shift_map(masks{i}, 16);
    sbj1 = labels(i);
    for j = i+1:n % 1:n
        m2 = masks{j};
        if isempty(m2)
            continue;
        end
        fv2 = feats{j};
        [s_mat(j, i), r_mat(j, i)] = ...
            cal_hd(fv1s, m1s, fv2, m2);
        
        if sbj1 == labels(j)
            idx_mat(j, i) = 1;
        else
            idx_mat(j, i) = -1;
        end
    end
    fprintf('%d\n', i);
end

g_scores = s_mat(idx_mat == 1 & r_mat > 0.2);
i_scores = s_mat(idx_mat == -1 & r_mat > 0.2);