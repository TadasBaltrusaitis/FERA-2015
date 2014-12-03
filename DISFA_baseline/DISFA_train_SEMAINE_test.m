clear 
SEMAINE_dir = 'I:\datasets\FERA_2015\Semaine\SEMAINE-Sessions\';

% train_recs = {'rec1', 'rec12', 'rec14', 'rec19', 'rec23', 'rec25', 'rec37', 'rec39', 'rec43', 'rec45', 'rec48', 'rec50', 'rec52', 'rec54', 'rec56', 'rec60'};
% devel_recs = {'rec9', 'rec13', 'rec15', 'rec20', 'rec24', 'rec26', 'rec38', 'rec42', 'rec44', 'rec46', 'rec49', 'rec51', 'rec53', 'rec55', 'rec58'};

% Issues with the following recordings 9, 50, 56, so they are omitted for
% now
train_recs = {'rec1', 'rec12', 'rec14', 'rec19', 'rec23', 'rec25', 'rec37', 'rec39', 'rec43', 'rec45', 'rec48', 'rec52', 'rec54', 'rec60'};
devel_recs = {'rec13', 'rec15', 'rec20', 'rec24', 'rec26', 'rec38', 'rec42', 'rec44', 'rec46', 'rec49', 'rec51', 'rec53', 'rec55', 'rec58'};

to_test = devel_recs;

aus_to_test = [2, 12, 17, 25];

[labels_gt, valid_ids, vid_inds] = extract_SEMAINE_labels(SEMAINE_dir, to_test, aus_to_test);

% Extract our baseline C++ results
output_semaine = 'I:\datasets\FERA_2015\Semaine\processed_data\';

pred_aus = [4,12,25,26,1,2,5,6,9,15,17];

inds_to_use = [];

for i=1:numel(aus_to_test)

    inds_to_use = cat(1, inds_to_use, find(pred_aus == aus_to_test(i)));

end

labels_pred = cell(numel(to_test), 1);

for i=1:numel(to_test)   
    labels = dlmread([output_semaine, to_test{i}, '.au.txt'], ' ');
    labels_pred{i} = labels(vid_inds(i,1):vid_inds(i,2)-1, inds_to_use+1);
end

labels_all_gt = cat(1, labels_gt{:});
labels_all_pred = cat(1, labels_pred{:});

labels_bin_pred = labels_all_pred > 0.99;

% Some simple correlations
for i=1:numel(aus_to_test)
   c = corr(labels_all_gt(:,i), labels_all_pred(:,i)); 
   
   tp = sum(labels_all_gt(:,i) == 1 & labels_bin_pred(:,i) == 1);
   fp = sum(labels_all_gt(:,i) == 0 & labels_bin_pred(:,i) == 1);
   fn = sum(labels_all_gt(:,i) == 1 & labels_bin_pred(:,i) == 0);
   tn = sum(labels_all_gt(:,i) == 0 & labels_bin_pred(:,i) == 0);
   
   precision = tp/(tp+fp);
   recall = tp/(tp+fn);
   
   f1 = 2 * precision * recall / (precision + recall);
   
   fprintf('AU%d: corr - %.3f, precision - %.3f, recall - %.3f, F1 - %.3f\n', aus_to_test(i), c, precision, recall, f1);
end