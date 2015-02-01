function [data_train, labels_train, data_valid, labels_valid, raw_valid, PC, means_norm, stds_norm] = ...
    Prepare_HOG_AU_data_generic_dynamic(train_users, au_train, rest_aus, hog_data_dir)

%% This should be a separate function?

input_train_label_files = cell(numel(train_users),1);
    
if(exist('F:/datasets/DISFA/', 'file'))
    root = 'F:/datasets/DISFA/';
elseif(exist('D:/Databases/DISFA/', 'file'))        
    root = 'D:/Databases/DISFA/';
elseif(exist('Z:/datasets/DISFA/', 'file'))        
    root = 'Z:/Databases/DISFA/';
elseif(exist('E:/datasets/DISFA/', 'file'))        
    root = 'E:/datasets/DISFA/';
elseif(exist('C:/tadas/DISFA/', 'file'))        
    root = 'C:/tadas/DISFA/';
elseif(exist('D:\datasets\face_datasets\DISFA/', 'file'))        
    root = 'D:\datasets\face_datasets\DISFA/';
else
    fprintf('DISFA location not found (or not defined)\n'); 
end    
    
% This is for loading the labels
for i=1:numel(train_users)   
    input_train_label_files{i} = [root, '/ActionUnit_Labels/', train_users{i}, '/', train_users{i}];
end

% First extracting the labels
[train_geom_data] = Read_geom_files_dynamic(train_users, hog_data_dir);

% Reading in the HOG data
[train_appearance_data, tracked_inds_hog, vid_ids_train] = Read_HOG_files_dynamic(train_users, hog_data_dir);

train_appearance_data = cat(2, train_appearance_data, train_geom_data);

% Getting the indices describing the splits (full version)
[training_inds, valid_inds, split] = construct_indices(vid_ids_train, train_users);

% Extracting the labels
labels_train = extract_au_labels(input_train_label_files, au_train);

% can now extract the needed training labels (do not rebalance validation
% data)
labels_valid = labels_train(valid_inds);

% make sure the same number of positive and negative samples is taken
pos_count = sum(labels_train(training_inds) > 0);
neg_count = sum(labels_train(training_inds) == 0);

inds_train = 1:size(labels_train,1);
neg_samples = inds_train(labels_train == 0 & training_inds);
to_rem = round(neg_count -  pos_count);
neg_samples_to_rem = neg_samples(round(linspace(1, size(neg_samples,2), to_rem)));

% Get rid of non tracked frames
training_inds = true(size(labels_train,1),1);
training_inds(valid_inds) = false;
training_inds(~tracked_inds_hog) = false;
training_inds(neg_samples_to_rem) = false;

labels_train = labels_train(training_inds);

% normalise the data
pca_file = '../../pca_generation/generic_face_rigid.mat';
load(pca_file);
     
PC_n = zeros(size(PC)+size(train_geom_data, 2));
PC_n(1:size(PC,1), 1:size(PC,2)) = PC;
PC_n(size(PC,1)+1:end, size(PC,2)+1:end) = eye(size(train_geom_data, 2));
PC = PC_n;

means_norm = cat(2, means_norm, zeros(1, size(train_geom_data,2)));
stds_norm = cat(2, stds_norm, ones(1, size(train_geom_data,2)));

% Grab all data for validation as want good params for all the data
raw_valid = train_appearance_data(valid_inds,:);

valid_appearance_data = bsxfun(@times, bsxfun(@plus, train_appearance_data(valid_inds,:), -means_norm), 1./stds_norm);
train_appearance_data = bsxfun(@times, bsxfun(@plus, train_appearance_data(training_inds,:), -means_norm), 1./stds_norm);

data_train = train_appearance_data * PC;
data_valid = valid_appearance_data * PC;

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