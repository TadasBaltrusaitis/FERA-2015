clear 
addpath('../data extraction/');
find_BP4D;

to_test = devel_recs;

aus_to_test = [1, 2, 4, 6, 12, 15, 17];

[labels_gt, valid_ids, vid_inds, filenames] = extract_BP4D_labels(BP4D_dir, to_test, aus_to_test);

% Extract our baseline C++ results
output_bp4d = 'I:\datasets\FERA_2015\BP4D\processed_data\';

%% Predict using the DISFA trained models (static)

addpath('../BP4D_baseline/');
labels_pred = cell(numel(labels_gt), 1);
labels_all_pred = [];

[raw_devel, ~, ~] = Read_HOG_files(devel_recs, [hog_data_dir, '/devel/']);
raw_devel_geom = Read_geom_files(devel_recs, [hog_data_dir, '/devel/']);
raw_devel = cat(2, raw_devel, raw_devel_geom);

load('../pca_generation/generic_face_rigid.mat');

PC_n = zeros(size(PC)+size(raw_devel_geom,2));
PC_n(1:size(PC,1), 1:size(PC,2)) = PC;
PC_n(size(PC,1)+1:end, size(PC,2)+1:end) = eye(size(raw_devel_geom, 2));
PC = PC_n;

means_norm = cat(2, means_norm, zeros(1, size(raw_devel_geom,2)));
stds_norm = cat(2, stds_norm, ones(1, size(raw_devel_geom,2)));
% Reading in the HOG data (of only relevant frames)


for i=1:numel(aus_to_test)   

    % load the appropriate model from the trained DISFA files
    model_file = sprintf('../DISFA_baseline/training/paper_res/AU_%d_static.mat', aus_to_test(i));
    load(model_file);
    
    % perform prediction with the model file
    % Go from raw data to the prediction
    w = model.w(1:end-1)';
    b = model.w(end);

    svs = bsxfun(@times, PC, 1./stds_norm') * w;

    % Attempt own prediction
    preds_mine = bsxfun(@plus, raw_devel, -means_norm) * svs + b;
    
    l1_inds = preds_mine > 0;
    l2_inds = preds_mine <= 0;
    preds_mine(l1_inds) = model.Label(1);
    preds_mine(l2_inds) = model.Label(2);
     
    labels_all_pred = cat(2, labels_all_pred, preds_mine);
    
end

%%
labels_all_gt = cat(1, labels_gt{:});

% Some simple correlations
for i=1:numel(aus_to_test)
   
   tp = sum(labels_all_gt(:,i) == 1 & labels_all_pred(:,i) == 1);
   fp = sum(labels_all_gt(:,i) == 0 & labels_all_pred(:,i) == 1);
   fn = sum(labels_all_gt(:,i) == 1 & labels_all_pred(:,i) == 0);
   tn = sum(labels_all_gt(:,i) == 0 & labels_all_pred(:,i) == 0);
   
   precision = tp/(tp+fp);
   recall = tp/(tp+fn);
   
   f1 = 2 * precision * recall / (precision + recall);
   
   fprintf('AU%d: precision - %.3f, recall - %.3f, F1 - %.3f\n', aus_to_test(i), precision, recall, f1);
end