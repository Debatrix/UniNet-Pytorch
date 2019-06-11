function [ d, mask_rate ] = cal_hd( f1s, m1s, f2, m2 )

d = Inf;
n = length(f1s);

for i = 1:n
    cmask = m1s{i} & m2;
    nb = nnz(cmask);
    diff = xor(f1s{i}, f2);
    d_tmp = nnz(diff(cmask)) / nb;
    if d_tmp < d
        d = d_tmp;
        mask_rate = nb/numel(cmask);
    end
end

end

