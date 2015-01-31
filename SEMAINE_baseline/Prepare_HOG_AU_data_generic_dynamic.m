function [data_train, labels_train, data_devel, labels_devel, raw_devel, PC, means_norm, stds_norm] = ...
    Prepare_HOG_AU_data_generic_dynamic(train_users, devel_users, au_train, rest_aus, semaine_dir, hog_data_dir, pca_file)

%%
addpath(genpath('../data extraction/'));

% First extracting the labels
[ labels_train, valid_ids_train, vid_ids_train ] = extract_SEMAINE_labels(semaine_dir, train_users, au_train);

% Reading in the HOG data (of only relevant frames)
[train_appearance_data, valid_ids_train_hog, vid_ids_train_string] = Read_HOG_files_dynamic(train_users, vid_ids_train, [hog_data_dir, '/train/']);

[train_geom_data] = Read_geom_files_dynamic(train_users, vid_ids_train, [hog_data_dir, '/train/']);

% Subsample the data to make training quicker
labels_train = cat(1, labels_train{:});
valid_ids_train = logical(cat(1, valid_ids_train{:}));

reduced_inds = true(size(labels_train,1),1);

% Remove invalid ids based on CLM failing or AU not being labelled
reduced_inds(~valid_ids_train) = false;
reduced_inds(~valid_ids_train_hog) = false;

labels_train = labels_train(reduced_inds,:);
train_appearance_data = train_appearance_data(reduced_inds,:);
train_geom_data = train_geom_data(reduced_inds,:);
vid_ids_train_string = vid_ids_train_string(reduced_inds,:);

%% Extract devel data

% First extracting the labels
[ labels_devel, valid_ids_devel, vid_ids_devel ] = extract_SEMAINE_labels(semaine_dir, devel_users, au_train);

% Reading in the HOG data (of only relevant frames)
[devel_appearance_data, valid_ids_devel_hog, vid_ids_devel_string] = Read_HOG_files_dynamic(devel_users, vid_ids_devel, [hog_data_dir, '/devel/']);

[devel_geom_data] = Read_geom_files_dynamic(devel_users, vid_ids_devel, [hog_data_dir, '/devel/']);

labels_devel = cat(1, labels_devel{:});

% normalise the data
load(pca_file);
     
% Grab all data for validation as want good params for all the data
raw_devel = cat(2, devel_appearance_data, devel_geom_data);

devel_appearance_data = bsxfun(@times, bsxfun(@plus, devel_appearance_data, -means_norm), 1./stds_norm);
train_appearance_data = bsxfun(@times, bsxfun(@plus, train_appearance_data, -means_norm), 1./stds_norm);

data_train = train_appearance_data * PC;
data_devel = devel_appearance_data * PC;

data_train = cat(2, data_train, train_geom_data);
data_devel = cat(2, data_devel, devel_geom_data);

PC_n = zeros(size(PC)+size(train_geom_data, 2));
PC_n(1:size(PC,1), 1:size(PC,2)) = PC;
PC_n(size(PC,1)+1:end, size(PC,2)+1:end) = eye(size(train_geom_data, 2));
PC = PC_n;

means_norm = cat(2, means_norm, zeros(1, size(train_geom_data,2)));
stds_norm = cat(2, stds_norm, ones(1, size(train_geom_data,2)));

end