clear 
BP4D_dir = 'I:\datasets\FERA_2015\BP4D\AUCoding\AUCoding\';
% Issues with the following recordings 9, 50, 56, so they are omitted for
% now
train_recs = {'F001', 'F003', 'F005', 'F007', 'F009', 'F011', 'F013', 'F015', 'F017', 'F019', 'F021', 'F023', 'M001', 'M003', 'M005', 'M007', 'M009', 'M011', 'M013', 'M015' 'M017'};

devel_recs = {'F002', 'F004', 'F006', 'F008', 'F010', 'F012', 'F014', 'F016', 'F018', 'F020', 'F022', 'M002', 'M004', 'M006', 'M008', 'M010', 'M012', 'M014', 'M016', 'M018'};

to_test = devel_recs;

aus_to_test = [1, 2, 4, 6, 12, 15, 17];

[labels_gt, valid_ids, vid_inds, filenames] = extract_BP4D_labels(BP4D_dir, to_test, aus_to_test);

% Extract our baseline C++ results
output_bp4d = 'I:\datasets\FERA_2015\BP4D\processed_data\';

pred_aus = [4,12,25,26,1,2,5,6,9,15,17];

inds_to_use = [];

for i=1:numel(aus_to_test)

    inds_to_use = cat(1, inds_to_use, find(pred_aus == aus_to_test(i)));

end

labels_pred = cell(numel(labels_gt), 1);

for i=1:numel(filenames)   
    labels = dlmread([output_bp4d, filenames{i}, '.au.txt'], ' ');
    labels_pred{i} = labels(:, inds_to_use+1);
    
    if(size(labels_pred{i},1) ~= size(labels_gt{i},1))
       fprintf('Something wrong at:%s\n',  filenames{i});
    end
    
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