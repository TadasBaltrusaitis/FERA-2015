function [data_train, labels_train, data_devel, labels_devel, raw_devel, PC, means_norm, stds_norm] = ...
    Prepare_HOG_AU_data_generic(train_users, devel_users, au_train, rest_aus, semaine_dir, hog_data_dir)

%%
addpath(genpath('../data extraction/'));

all_aus = [au_train, rest_aus];

% First extracting the labels
[ labels_train, valid_ids_train, vid_ids_train ] = extract_SEMAINE_labels(semaine_dir, train_users, all_aus);

% Reading in the HOG data (of only relevant frames)
[train_appearance_data, vid_ids_train_string] = Read_HOG_files(train_users, vid_ids_train, hog_data_dir);

%%

% Reading in the HOG data (of only relevant frames)
[devel_appearance_data, vid_ids_devel_string] = Read_HOG_files(devel_users, vid_ids_train, hog_data_dir);

% First extracting the labels
[ labels_devel, valid_ids_devel, vid_ids_devel_string ] = extract_SEMAINE_labels(semaine_dir, devel_users, all_aus);

% Getting the indices describing the splits (full version)
[training_inds, valid_inds, split] = construct_indices(vid_ids_train, train_users);

% Getting the rebalanced training and validation indices and data
[inds_train_rebalanced, inds_valid_rebalanced, ~] = construct_balanced(labels_train, training_inds, valid_inds, rest_aus, input_train_label_files);

% can now extract the needed training labels (do not rebalance validation
% data)
labels_devel = labels_train(valid_inds);
labels_train = labels_train(inds_train_rebalanced);

% normalise the data
pca_file = '../../data_extraction/generic_face_rigid.mat';
load(pca_file);
     
% Grab all data for validation as want good params for all the data
raw_devel = train_appearance_data(valid_inds,:);

valid_appearance_data = bsxfun(@times, bsxfun(@plus, train_appearance_data(valid_inds,:), -means_norm), 1./stds_norm);
train_appearance_data = bsxfun(@times, bsxfun(@plus, train_appearance_data(inds_train_rebalanced,:), -means_norm), 1./stds_norm);

data_train = train_appearance_data * PC;
data_devel = valid_appearance_data * PC;

end

function [training_inds, valid_inds, split] = construct_indices(video_inds, train_users)

    % Randomise the training and validation users TODO this makes it worse?
%     train_users = train_users(randperm(numel(train_users)));

    % extract these separately so as to guarantee person independent splits for
    % validation
    split = round(2*numel(train_users)/3);

    users_train = train_users(1:split);

    training_inds = false(size(video_inds));
    for i=1:numel(users_train)
        user_ind = strcmp(video_inds,  users_train(i));
        training_inds = training_inds | user_ind;
    end
    
    users_valid = train_users(split+1:end);
    valid_inds = false(size(video_inds));
    for i=1:numel(users_valid)
        user_ind = strcmp(video_inds,  users_valid(i));
        valid_inds = valid_inds | user_ind;
    end        
end

function [inds_train_rebalanced, inds_valid_rebalanced, sample_length] = construct_balanced(labels_all, training_inds, valid_inds,...
                                                                            rest_aus, input_labels_files)
    
    all_subs = 1:size(labels_all,1);
                                                                        
    labels_train = labels_all(training_inds);
    labels_valid = labels_all(valid_inds);
    
    labels_other_au = zeros(size(labels_all,1), numel(rest_aus));

    % This is used to pick up activity of other AUs for a more 'interesting'
    % data split and not only neutral expressions for negative samples    
    for i=1:numel(rest_aus)
        labels_other_au(:,i) = extract_au_labels(input_labels_files, rest_aus(i));
    end
    
    labels_other_au_train = labels_other_au(training_inds,:);
    labels_other_au_valid = labels_other_au(valid_inds,:);
    
    [ inds_train_rebalanced, sample_length ] = extract_events( labels_train, labels_other_au_train );
    [ inds_valid_rebalanced ] = extract_events( labels_valid, labels_other_au_valid, sample_length );
    
    % Need to move from indices in labels_train space to indices in
    % original labels_all space
    sub_train = all_subs(training_inds);
    inds_train_rebalanced = sub_train(inds_train_rebalanced);
    
    sub_valid = all_subs(valid_inds);
    inds_valid_rebalanced = sub_valid(inds_valid_rebalanced);
end