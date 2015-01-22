addpath('../data extraction/');

find_BP4D;

aus_BP4D = [6, 10, 12, 14, 17];

root = 'out_bp4d_all/';

[ labels_gt, valid_ids, vid_ids, filenames] = extract_BP4D_labels_intensity(BP4D_dir_int, devel_recs, aus_BP4D);

labels_pred = [];
for i=1:numel(filenames)
    lbl = dlmread([root, filenames{i}, '.au.reg.txt'], ' ', 0, 1)';
    assert(size(lbl,1) == size(labels_gt{i}, 1))
    labels_pred = cat(1, labels_pred, lbl);
end

labels_gt = cat(1, labels_gt{:});
labels_gt(labels_gt>5) = 0;
for a = 1:numel(aus_BP4D)
       
    c = corr(labels_gt(:,a), labels_pred(:,a));
    mse = mean((labels_gt(:,a) - labels_pred(:,a)).^2);
    fprintf('AU%d, correlation- %.3f, MSE - %.3f\n', aus_BP4D(a), c, mse);
    
end