function [data_train, labels_train, data_devel, labels_devel, raw_devel, PC, means_norm, stds_norm] = ...
    Prepare_HOG_AU_data_generic_geom(train_users, devel_users, au_train, bp4d_dir, hog_data_dir, pca_file)

%%
addpath(genpath('../data extraction/'));

% First extracting the labels
[ labels_train, valid_ids_train, vid_ids_train ] = extract_BP4D_labels(bp4d_dir, train_users, au_train);

au_other = setdiff([1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23], au_train);
[ labels_other, ~, ~ ] = extract_BP4D_labels(bp4d_dir, train_users, au_other);
labels_other = cat(1, labels_other{:});

[train_geom_data, valid_ids_train_geom] = Read_geom_files(train_users, [hog_data_dir, '/train/']);

% Subsample the data to make training quicker
labels_train = cat(1, labels_train{:});
valid_ids_train = logical(cat(1, valid_ids_train{:}));

reduced_inds = false(size(labels_train,1),1);
reduced_inds(labels_train == 1) = true;

% make sure the same number of positive and negative samples is taken
pos_count = sum(labels_train == 1);
neg_count = sum(labels_train == 0);

num_other = floor(pos_count / (size(labels_other, 2)));

inds_all = 1:size(labels_train,1);

for i=1:size(labels_other, 2)+1
   
    if(i > size(labels_other, 2))
        % fill the rest with a proportion of neutral
        inds_other = inds_all(sum(labels_other,2)==0 & ~labels_train );   
        num_other_i = min(numel(inds_other), pos_count - sum(labels_train(reduced_inds,:)==0));     
    else
        % take a proportion of each other AU
        inds_other = inds_all(labels_other(:, i) & ~labels_train );      
        num_other_i = min(numel(inds_other), num_other);        
    end
    inds_other_to_keep = inds_other(round(linspace(1, numel(inds_other), num_other_i)));
    reduced_inds(inds_other_to_keep) = true;
    
end

% Remove invalid ids based on CLM failing or AU not being labelled
reduced_inds(~valid_ids_train) = false;
reduced_inds(~valid_ids_train_geom) = false;

labels_other = labels_other(reduced_inds, :);
labels_train = labels_train(reduced_inds,:);
train_geom_data = train_geom_data(reduced_inds,:);

%% Extract devel data

% First extracting the labels
[ labels_devel, valid_ids_devel, vid_ids_devel ] = extract_BP4D_labels(bp4d_dir, devel_users, au_train);

% Reading in the HOG data (of only relevant frames)
[devel_geom_data, valid_ids_devel_geom] = Read_geom_files(devel_users, [hog_data_dir, '/devel/']);

labels_devel = cat(1, labels_devel{:});

% normalise the data
% load(pca_file);
     
% PC_n = zeros(size(PC)+size(train_geom_data, 2));
% PC_n(1:size(PC,1), 1:size(PC,2)) = PC;
% PC_n(size(PC,1)+1:end, size(PC,2)+1:end) = eye(size(train_geom_data, 2));
PC = eye(size(train_geom_data, 2));

means_norm =  mean(train_geom_data);
stds_norm = std(train_geom_data);

means_norm(:) = 0;
stds_norm(:) = 1;

% Grab all data for validation as want good params for all the data
raw_devel = devel_geom_data;

devel_geom_data = bsxfun(@times, bsxfun(@plus, devel_geom_data, -means_norm), 1./stds_norm);
train_geom_data = bsxfun(@times, bsxfun(@plus, train_geom_data, -means_norm), 1./stds_norm);

% TODO rem
% % Pick a thousand of samples
% % rng(0);
% % samples = randperm(size(labels_train,1));
% % max_s = min(3000, size(labels_train,1));
% % 
% % samples = samples(1:max_s);
% 
% opts = statset('display','iter');
% 
% fs = sequentialfs(@critfun, train_geom_data(1:10:end,:), labels_train(1:10:end,:), 'cv', 3, 'options', opts, 'direction', 'forward');
% PC(~fs,:) = 0;
% save(sprintf('au_%d_feats.mat', au_train), 'fs');
data_train = train_geom_data * PC;
data_devel = devel_geom_data * PC;

end

function dev = critfun(X,Y, X2, Y2)

    comm = sprintf('-s 1 -B 1 -q');
    model = train(Y, sparse(X), comm);

    w = model.w(1:end-1)';
    b = model.w(end);

    % Attempt own prediction
    prediction = X2 * w + b;
    l1_inds = prediction > 0;
    l2_inds = prediction <= 0;
    prediction(l1_inds) = model.Label(1);
    prediction(l2_inds) = model.Label(2);
 
    %[prediction, a, actual_vals] = predict(test_labels, test_samples, model);           
    
    tp = sum(Y2 == 1 & prediction == 1);
    fp = sum(Y2 == 0 & prediction == 1);
    fn = sum(Y2 == 1 & prediction == 0);
%     tn = sum(Y2 == 0 & prediction == 0);

    precision = tp/(tp+fp);
    recall = tp/(tp+fn);

    f1 = 2 * precision * recall / (precision + recall);

%     result = corr(test_labels, prediction);
%     fprintf('F1:%.3f\n', f1);
    if(isnan(f1))
        f1 = 0;
    end
    dev = -f1;
end