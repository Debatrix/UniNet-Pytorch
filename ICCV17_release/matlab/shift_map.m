function feats = shift_map(f, shifts, step)

if nargin < 3
    step = 1;
end

[w, h] = size(f);

feats = cell(2*shifts+1, 1);

for s = -shifts : shifts
    f2 = f;
    s2 = step*s;
    if s < 0
        f2(1:w+s2, :) = f(1-s2:w, :);
        f2(w+s2+1:w, :) = f(1:-s2, :);
    else
        f2(1:s2, :) = f(w-s2+1:w, :);
        f2(s2+1:w, :) = f(1:w-s2, :);
    end
    feats{s+shifts+1} = f2;
end

end