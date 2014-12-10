clear 
addpath('../data extraction/');
find_BP4D;

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

%% Predict using the DISFA trained models (static)

addpath('../BP4D_baseline/');
labels_pred = cell(numel(labels_gt), 1);
labels_all_pred = [];

load('../pca_generation/generic_face_rigid.mat');

% Reading in the HOG data (of only relevant frames)
[raw_devel, ~, ~] = Read_HOG_files(devel_recs, [hog_data_dir, '/devel/']);

for i=1:numel(aus_to_test)   

    % load the appropriate model from the trained DISFA files
    model_file = sprintf('../DISFA_baseline/training/trained/AU_%d_static.mat', aus_to_test(i));
    load(model_file);
    
    % perform prediction with the model file
    % Go from raw data to the prediction
    w = model.w(1:end-1)';
    b = model.w(end);

    svs = bsxfun(@times, PC, 1./stds_norm') * w;

    % Attempt own prediction
    preds_mine = bsxfun(@plus, raw_devel, -means_norm) * svs + b;
    preds_mine(preds_mine <0) = 0;
    preds_mine(preds_mine >5) = 5;
    
    labels_all_pred = cat(2, labels_all_pred, preds_mine);
    
end

%%
labels_all_gt = cat(1, labels_gt{:});

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