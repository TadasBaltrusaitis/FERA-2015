function [data_train, labels_train, data_devel, labels_devel, raw_devel, PC, means_norm, stds_norm] = ...
    Prepare_HOG_AU_data_generic_dynamic_geom(train_users, devel_users, au_train, rest_aus, semaine_dir, hog_data_dir, pca_file)

%%
addpath(genpath('../data extraction/'));

% First extracting the labels
[ labels_train, valid_ids_train, vid_ids_train ] = extract_SEMAINE_labels(semaine_dir, train_users, au_train);

[ labels_other, ~, ~ ] = extract_SEMAINE_labels(semaine_dir, train_users, rest_aus);
labels_other = cat(1, labels_other{:});

[train_geom_data, valid_ids_train_geom] = Read_geom_files_dynamic(train_users, vid_ids_train, [hog_data_dir, '/train/']);

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
[ labels_devel, valid_ids_devel, vid_ids_devel ] = extract_SEMAINE_labels(semaine_dir, devel_users, au_train);

[devel_geom_data, valid_ids_devel_geom] = Read_geom_files_dynamic(devel_users, vid_ids_devel, [hog_data_dir, '/devel/']);

labels_devel = cat(1, labels_devel{:});

means_norm =  mean(train_geom_data);
stds_norm = std(train_geom_data);

% Grab all data for validation as want good params for all the data
raw_devel = devel_geom_data;

% devel_geom_data = bsxfun(@times, bsxfun(@plus, devel_geom_data, -means_norm), 1./stds_norm);
% train_geom_data = bsxfun(@times, bsxfun(@plus, train_geom_data, -means_norm), 1./stds_norm);
% Grab all data for validation as want good params for all the data
% raw_devel = devel_geom_data;

data_train = train_geom_data;
data_devel = devel_geom_data;

PC =  eye(size(train_geom_data, 2));

means_norm = zeros(1, size(train_geom_data,2));
stds_norm = ones(1, size(train_geom_data,2));

end