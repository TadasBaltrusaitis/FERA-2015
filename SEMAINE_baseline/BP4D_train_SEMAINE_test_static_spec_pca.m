clear 
addpath(genpath('../data extraction/'));
find_SEMAINE;

% train_recs = {'rec1', 'rec12', 'rec14', 'rec19', 'rec23', 'rec25', 'rec37', 'rec39', 'rec43', 'rec45', 'rec48', 'rec50', 'rec52', 'rec54', 'rec56', 'rec60'};
% devel_recs = {'rec9', 'rec13', 'rec15', 'rec20', 'rec24', 'rec26', 'rec38', 'rec42', 'rec44', 'rec46', 'rec49', 'rec51', 'rec53', 'rec55', 'rec58'};

% Issues with the following recordings 9, 50, 56, so they are omitted for
% now
train_recs = {'rec1', 'rec12', 'rec14', 'rec19', 'rec23', 'rec25', 'rec37', 'rec39', 'rec43', 'rec45', 'rec48', 'rec52', 'rec54', 'rec60'};
devel_recs = {'rec13', 'rec15', 'rec20', 'rec24', 'rec26', 'rec38', 'rec42', 'rec44', 'rec46', 'rec49', 'rec51', 'rec53', 'rec55', 'rec58'};

to_test = devel_recs;

aus_to_test = [2, 12, 17];

[labels_gt, valid_ids, vid_inds] = extract_SEMAINE_labels(SEMAINE_dir, to_test, aus_to_test);

%% Predict using the DISFA trained models (static)

addpath('../SEMAINE_baseline/');
labels_pred = cell(numel(labels_gt), 1);
labels_all_pred = [];

load('../pca_generation/bp4d_model.mat');

% Reading in the HOG data (of only relevant frames)
[raw_devel, ~, ~] = Read_HOG_files(devel_recs, vid_inds, [hog_data_dir, '/devel/']);

%%
for i=1:numel(aus_to_test)   

    % load the appropriate model from the trained DISFA files
    model_file = sprintf('../BP4D_baseline/trained/AU_%d_static_bp4d_pca.mat', aus_to_test(i));
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