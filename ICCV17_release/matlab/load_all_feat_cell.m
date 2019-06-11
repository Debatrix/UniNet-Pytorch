function [ feats, labels ] = load_all_feat_cell( folder, max_num )
%LOAD_ALL_FEAT Summary of this function goes here
%   Detailed explanation goes here

if nargin < 2
    max_num = Inf;
end

files = dir([folder, '/*.mat']);
n = min(max_num, length(files));

feats = cell(n, 1);
labels = zeros(n, 1);

for i = 1:n
    d = load([folder, '/', files(i).name]);
    feats{i} = d.output;
    labels(i) = parse_label(files(i).name);
end

    function label = parse_label(str)
        l = length(str);
        p1 = 0;
        for p = 1:l
            if p1 == 0 && str(p) >= '0' && str(p) <= '9'
                p1 = p;
            end
            if p1 > 0 && (str(p) < '0' || str(p) > '9')
                p2 = p-1;
                break;
            end
        end
        label = str2double(str(p1:p2));
    end

end
