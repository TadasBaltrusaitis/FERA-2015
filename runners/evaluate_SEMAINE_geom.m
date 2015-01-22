addpath(genpath('../data extraction/'));

find_SEMAINE;

aus_SEMAINE = [2 12 17 25 28 45];

root = 'out_SEMAINE_dynamic_geom/';

[ labels_gt, valid_ids, vid_ids] = extract_SEMAINE_labels(SEMAINE_dir, devel_recs, aus_SEMAINE);
labels_gt = cat(1, labels_gt{:});

labels_pred = [];
for i=1:numel(devel_recs)
    lbl = dlmread([root, devel_recs{i}, '.au.txt'], ' ', 0, 1)';
    lbl = lbl(2:end,:);
    labels_pred = cat(1, labels_pred, lbl);
end

tp = sum(labels_gt == 1 & labels_pred == 1);
fp = sum(labels_gt == 0 & labels_pred == 1);
fn = sum(labels_gt == 1 & labels_pred == 0);
tn = sum(labels_gt == 0 & labels_pred == 0);

precision = tp./(tp+fp);
recall = tp./(tp+fn);

f1 = 2 * precision .* recall ./ (precision + recall);

for a = 1:numel(aus_SEMAINE)
       
    fprintf('AU%d, Precision - %.3f, Recall - %.3f, F1 - %.3f\n', aus_SEMAINE(a), precision(a), recall(a), f1(a));
    
end