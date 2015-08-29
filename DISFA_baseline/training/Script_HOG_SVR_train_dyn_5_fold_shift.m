% Change to your downloaded location
addpath('C:\liblinear\matlab')

num_test_folds = 5;

%% load shared definitions and AU data
shared_defs;

% Set up the hyperparameters to be validated
hyperparams.c = 10.^(-7:1:3);
hyperparams.p = 10.^(-2);

hyperparams.validate_params = {'c', 'p'};

% Set the training function
svr_train = @svr_train_linear_shift_learn;
    
% Set the test function (the first output will be used for validation)
svr_test = @svr_test_linear_shift_learn;

%%
for a=2:numel(aus)
    
    au = aus(a);
    
    prediction_all = [];
    test_all = [];
    
    for t=1:num_test_folds

        rest_aus = setdiff(all_aus, au);        

        % load the training and testing data for the current fold
        [~, ~, test_samples, test_labels, ~, ~, ~, ~, test_ids, success_test] = Prepare_HOG_AU_data_generic_dynamic(users, au, rest_aus, hog_data_dir, t, 1/num_test_folds);

        users_train = setdiff(users, unique(test_ids));

        % need to split the rest
        [train_samples, train_labels, valid_samples, valid_labels, ~, PC, means, scaling, valid_ids, success_valid] = Prepare_HOG_AU_data_generic_dynamic(users_train, au, rest_aus, hog_data_dir);

        train_samples = sparse(train_samples);
        valid_samples = sparse(valid_samples);
        test_samples = sparse(test_samples);

        hyperparams.valid_samples = valid_samples;
        hyperparams.valid_labels = valid_labels;
        hyperparams.vid_ids = valid_ids;
        hyperparams.success = success_valid;
    
        %% Cross-validate here                
        [ best_params, ~ ] = validate_grid_search(svr_train, svr_test, false, train_samples, train_labels, valid_samples, valid_labels, hyperparams);

        model = svr_train(train_labels, train_samples, best_params);        
        model.vid_ids = test_ids;
        model.success = success_test;
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
    name = sprintf('5_fold_shift/AU_%d_dyn_shift.mat', au);

    [ accuracies, F1s, corrs, ccc, rms, classes ] = evaluate_classification_results( prediction_all, test_all );    

    save(name, 'model', 'accuracies', 'F1s', 'corrs', 'ccc', 'rms', 'prediction_all', 'test_all');        

end
