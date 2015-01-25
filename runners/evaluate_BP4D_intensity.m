addpath('../data extraction/');

find_BP4D;

aus_BP4D = [6, 10, 12, 14, 17];

root = 'out_bp4d_all_latest/';

[ labels_gt, valid_ids, vid_ids, filenames] = extract_BP4D_labels_intensity(BP4D_dir_int, devel_recs, aus_BP4D);

labels_pred = [];
labels_pred_seg = [];
for i=1:numel(filenames)
    lbl = dlmread([root, filenames{i}, '.au.reg.txt'], ' ', 0, 1)';
    lbl2 = dlmread([root, filenames{i}, '.au.reg.seg.txt'], ' ', 0, 1)';
    if(size(lbl,1) > size(labels_gt{i}, 1))
        lbl = lbl(1:size(labels_gt{i}, 1), :);
        fprintf('Something wrong at %s \n', filenames{i})
    else
        assert(size(lbl,1) == size(labels_gt{i}, 1))
    end
    if(size(lbl2,1) > size(labels_gt{i}, 1))
        lbl2 = lbl2(1:size(labels_gt{i}, 1), :);
        fprintf('Something wrong at %s \n', filenames{i})
    else
        assert(size(lbl2,1) == size(labels_gt{i}, 1))
    end    
    
    labels_pred = cat(1, labels_pred, lbl);
    labels_pred_seg = cat(1, labels_pred_seg, lbl2);
end

labels_gt = cat(1, labels_gt{:});
labels_gt(labels_gt>5) = 0;
for a = 1:numel(aus_BP4D)
    
    segmented = labels_gt(:,a) >= 1;
    
    c = corr(labels_gt(:,a), labels_pred(:,a));
    mse = mean((labels_gt(:,a) - labels_pred(:,a)).^2);
    
    c_seg = corr(labels_gt(segmented,a), labels_pred_seg(segmented,a));
    mse_seg = mean((labels_gt(segmented,a) - labels_pred_seg(segmented,a)).^2);
    
    fprintf('AU%d, correlation- %.3f, MSE - %.3f corr seg %.3f MSE seg %.3f\n', aus_BP4D(a), c, mse, c_seg, mse_seg);
    
end