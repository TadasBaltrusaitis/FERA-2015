% Change to your downloaded location
addpath('C:\liblinear\matlab')
addpath('../data extraction/');

num_test_folds = 5;

%% load shared definitions and AU data
shared_defs;

% Set up the hyperparameters to be validated
hyperparams.c = 10.^(-7:0.5:1);
hyperparams.e = 10.^(-3);

hyperparams.validate_params = {'c', 'e'};

% Set the training function
svm_train = @svm_train_linear;
    
% Set the test function (the first output will be used for validation)
svm_test = @svm_test_linear;

pca_loc = '../pca_generation/generic_face_rigid.mat';
all_recs_BP4D = cat(2, train_recs, devel_recs);
hog_data_dir_BP4D = hog_data_dir;

aus = [1,2,4,15];
%%
for a=1:numel(aus)
    
    predictions_all = [];
    test_labels_all = [];
    
    au = aus(a);

    rest_aus = setdiff(all_aus, au);        

    [train_recs, valid_BP4D_recs] = get_balanced_fold(BP4D_dir, all_recs_BP4D, au, 1/4);

    % load the training and testing data for the current fold
    [train_samples, train_labels, valid_samples, valid_labels, ~, PC, means, scaling] = Prepare_HOG_AU_data_generic(train_recs, valid_BP4D_recs, au, BP4D_dir, hog_data_dir_BP4D, pca_loc);

    train_samples = sparse(train_samples);
    valid_samples = sparse(valid_samples);

    %% Cross-validate here                
    [ best_params, ~ ] = validate_grid_search_no_par(svm_train, svm_test, false, train_samples, train_labels, valid_samples, valid_labels, hyperparams);
    model = svm_train(train_labels, train_samples, best_params);        
    
    %% Now apply that model on DISFA
    for t = 1:num_test_folds

        % Test on DISFA
        find_DISFA;        
        od = cd('../DISFA_baseline/training/');
        test_folds = get_test_folds(num_test_folds, users);
        test_users = test_folds{t};
        train_users = setdiff(users, test_users); 
        all_disfa = [1,2,4,5,6,9,12,15,17,20,25,26];
        rest_aus = setdiff(all_disfa, au);    
        [valid_samples_disfa, valid_labels_disfa, test_samples_disfa, test_labels_disfa] = Prepare_HOG_AU_data_generic(train_users, test_users, au, rest_aus, hog_data_dir);
        cd(od);
        
        % Fine-tune on DISFA
        
        model_d = model;
        % Convert to a regressor from the classifier
        prediction_DISFA_valid = valid_samples_disfa * model_d.w(1:end-1)' + model_d.w(end);

        train_DISFA = cat(2, ones(numel(prediction_DISFA_valid),1), prediction_DISFA_valid);
        p = train_DISFA \ valid_labels_disfa;
        
        model_d.w(1:end) = model_d.w(1:end) * p(2);
        model_d.w(end) = model_d.w(end) + p(1);        
        
        prediction_DISFA_valid = valid_samples_disfa * model_d.w(1:end-1)' + model_d.w(end);
        
        predictions_DISFA_test = test_samples_disfa * model_d.w(1:end-1)' + model_d.w(end);
        
        predictions_DISFA_test(predictions_DISFA_test < 0) = 0;
        predictions_DISFA_test(predictions_DISFA_test > 5) = 5;
        
        predictions_all = cat(1, predictions_all, predictions_DISFA_test);
        test_labels_all = cat(1, test_labels_all, test_labels_disfa);
        
    end

    name = sprintf('results_regression/AU_%d_static.mat', au);

    [ accuracies, F1s, corrs, ccc, rms, classes ] = evaluate_classification_results( predictions_all, test_labels_all );
    
    save(name, 'model', 'F1s', 'corrs', 'accuracies', 'ccc', 'rms', 'predictions_all', 'test_labels_all');
        
end


