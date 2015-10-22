function [train_users, dev_users] = get_balanced_fold(SEMAINE_dir, users, au, prop_test)

    % Extracting the labels
    [labels_train,~,vid_ids_train] = extract_SEMAINE_labels(SEMAINE_dir, users, au);
    
    counts = zeros(numel(users),1);
    for k=1:numel(users)
        counts(k) = sum(labels_train{k});
    end

    [sorted, inds] = sort(counts);

    dev_users = users(inds(1:round(1/prop_test):end));
    train_users = setdiff(users, dev_users);
end