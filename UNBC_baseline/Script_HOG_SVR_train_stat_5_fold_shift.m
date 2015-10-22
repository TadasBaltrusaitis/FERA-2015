% Change to your downloaded location
clear
addpath('C:\liblinear\matlab')

num_test_folds = 5;

%% load shared definitions and AU data
shared_defs;

% Set up the hyperparameters to be validated
hyperparams.c = 10.^(-7:1:3);
hyperparams.p = 10.^(-2);

hyperparams.validate_params = {'c', 'p'};

% Set the training function
svr_train = @svr_train_linear_shift;
    
% Set the test function (the first output will be used for validation)
svr_test = @svr_test_linear_shift;

users = cat(2, devel_recs, train_recs);

test_folds = get_test_folds(num_test_folds, users);

%%
for a=1:numel(aus)
    
    au = aus(a);
    
    prediction_all = [];
    test_all = [];
    
    for t=1:num_test_folds

        rest_aus = setdiff(all_aus, au);        

        % load the training and testing data for the current fold
        [~, ~, test_samples, test_labels, ~, ~, ~, ~, test_ids, test_success] = Prepare_HOG_AU_data({}, test_folds{t}, au, rest_aus, UNBC_dir, hog_data_dir, '../pca_generation/generic_face_rigid.mat');

        % create the training and validation data
        users_train = setdiff(users, unique(test_ids));        
        % make sure validation data's labels are balanced
        [users_train, users_valid] = get_balanced_fold(UNBC_dir, users_train, au, 1/4, 1);
        
        % need to split the rest
        [train_samples, train_labels, valid_samples, valid_labels, ~, PC, means, scaling, valid_ids, valid_success] = Prepare_HOG_AU_data(users_train, users_valid, au, rest_aus, UNBC_dir, hog_data_dir, '../pca_generation/generic_face_rigid.mat');

        train_samples = sparse(train_samples);
        valid_samples = sparse(valid_samples);
        test_samples = sparse(test_samples);

        %% Cross-validate here  
        hyperparams.success = valid_success;
        hyperparams.valid_samples = valid_samples;
        hyperparams.valid_labels = valid_labels;
        hyperparams.vid_ids = valid_ids;     
        
        [ best_params, ~ ] = validate_grid_search_no_par(svr_train, svr_test, false, train_samples, train_labels, valid_samples, valid_labels, hyperparams);

        model = svr_train(train_labels, train_samples, best_params);        
        model.success = test_success;
        model.vid_ids = test_ids;
        
        [~, prediction] = svr_test(test_labels, test_samples, model);

        prediction_all = cat(1, prediction_all, prediction);
        test_all = cat(1, test_all, test_labels);
        
        % Go from raw data to the prediction
%         w = model.w(1:end-1)';
%         b = model.w(end);
% 
%         svs = bsxfun(@times, PC, 1./scaling') * w;

        % Attempt own prediction
    %     preds_mine = bsxfun(@plus, raw_valid, -means) * svs + b;
    %     preds_mine(preds_mine <0) = 0;
    %     preds_mine(preds_mine >5) = 5;
    % 
    %     assert(norm(preds_mine - prediction) < 1e-8);

    %     name = sprintf('results_ideal/AU_%d_static_vv.dat', au);
    % 
    %     write_lin_dyn_svr(name, means, svs, b);
    end
    name = sprintf('5_fold_shift/AU_%d_stat_shift.mat', au);

    [ accuracies, F1s, corrs, ccc, rms, classes ] = evaluate_classification_results( prediction_all, test_all );    

    save(name, 'model', 'accuracies', 'F1s', 'corrs', 'rms', 'ccc', 'prediction_all', 'test_all');        

end


