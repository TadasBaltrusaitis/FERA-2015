function [data_train, labels_train, data_devel, labels_devel, raw_devel, PC, means_norm, stds_norm] = ...
    Prepare_HOG_AU_data_generic_dynamic_c_2(train_users, devel_users, au_train, rest_aus, semaine_dir, hog_data_dir, pca_file)

%%
addpath(genpath('../data extraction/'));

% First extracting the labels
[ labels_train, valid_ids_train, vid_ids_train ] = extract_SEMAINE_labels(semaine_dir, train_users, au_train);

[ labels_other, ~, ~ ] = extract_SEMAINE_labels(semaine_dir, train_users, rest_aus);
labels_other = cat(1, labels_other{:});

% Reading in the HOG data (of only relevant frames)
[train_appearance_data, valid_ids_train_hog, vid_ids_train_string] = Read_HOG_files_dynamic(train_users, vid_ids_train, [hog_data_dir, '/train/']);

[train_geom_data] = Read_geom_files_dynamic(train_users, vid_ids_train, [hog_data_dir, '/train/']);

% Subsample the data to make training quicker
labels_train = cat(1, labels_train{:});
valid_ids_train = logical(cat(1, valid_ids_train{:}));


% make sure the same number of positive and negative samples is taken
pos_count = sum(labels_train == 1);
neg_count = sum(labels_train == 0);

% first overample a bit
if(pos_count * 2 < neg_count)
    pos_count = pos_count * 2;
end

reduced_inds = false(size(labels_train,1),1);
reduced_inds(labels_train == 1) = true;

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
reduced_inds(~valid_ids_train_hog) = false;

labels_other = labels_other(reduced_inds, :);
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

% oversample a bit
[labels_train, data_train] = oversample(labels_train, data_train);

end

function [train_labels, train_samples] = oversample(train_labels, train_samples)
    
    num_neighbors = 40;    
    
    inds_train = 1:size(train_labels,1);
    pos_samples = inds_train(train_labels == 1);

    % do not produce more pos than neg
    extra_num = numel(pos_samples);

    train_labels_extra = ones(extra_num, 1);
    train_samples_extra = zeros(extra_num, size(train_samples, 2));

    pos_samples = full(train_samples(pos_samples,:));

    KDTreeMdl = KDTreeSearcher(pos_samples);

    extra_sampling = round(linspace(1, size(pos_samples,1), floor(extra_num / num_neighbors)));
    filled = 1;
    for i=extra_sampling

        % discard one as it will be self
        curr_sample = pos_samples(i,:);
        [idxMk, Dmk] = knnsearch(KDTreeMdl, curr_sample, 'k', 100);
        inds = randperm(100);
        inds = inds(1:num_neighbors);
        old_samples = pos_samples(idxMk(inds),:);

        weights = rand(num_neighbors, 1);

        new_samples = bsxfun(@times, old_samples, weights) + bsxfun(@times, curr_sample, 1 - weights);
        train_samples_extra(filled:filled+num_neighbors-1,:) = new_samples;
        filled = filled + num_neighbors;
    end
    
    train_labels = cat(1, train_labels, train_labels_extra(1:filled, :));
    train_samples = cat(1, train_samples, train_samples_extra(1:filled, :));        
    
end