function [train_users, dev_users] = get_balanced_fold(users, au, prop_test, offset)

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
    elseif(exist('D:\Datasets\DISFA/','file'))
        root = 'D:\Datasets\DISFA/';
    else
        fprintf('DISFA location not found (or not defined)\n');
    end    

    % This is for loading the labels
    for i=1:numel(users)   
        input_train_label_files{i} = [root, '/ActionUnit_Labels/', users{i}, '/', users{i}];
    end

    % Extracting the labels
    labels_train = extract_au_labels(input_train_label_files, au);

    counts = zeros(numel(users),1);
    for k=1:numel(users)
        counts(k) = sum(labels_train((k-1)*4844+1:k*4844));
    end

    [sorted, inds] = sort(counts);
    
    dev_users = users(inds(offset:round(1/prop_test):end));
    train_users = setdiff(users, dev_users);
    
    count_dev = 0;
    count_train = 0;
    for k=1:numel(users)
        if(any(strcmp(dev_users, users{k})))
            count_dev = count_dev + counts(k);
        else
            count_train = count_train + counts(k);
        end
        
    end
    fprintf('Mean train %.2f, mean test %.2f\n', count_train / numel(train_users), count_dev / numel(dev_users));
    
end