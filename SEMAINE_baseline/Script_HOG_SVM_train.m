function Script_HOG_SVM_train()

% Change to your downloaded location
addpath('C:\liblinear\matlab')

%% load shared definitions and AU data
shared_defs;

% Set up the hyperparameters to be validated
hyperparams.c = 10.^(-7:0.5:-2);
hyperparams.e = 10.^(-3);

hyperparams.validate_params = {'c', 'e'};

% Set the training function
svm_train = @svm_train_linear;
    
% Set the test function (the first output will be used for validation)
svm_test = @svm_test_linear;

pca_loc = '../pca_generation/generic_face_rigid.mat';
all_recs = cat(2,train_recs, devel_recs);

%%
for a=1:numel(aus)
    
    au = aus(a);
            
    rest_aus = setdiff(all_aus, au);        

    [train_recs, test_recs] = get_balanced_fold(SEMAINE_dir, all_recs, au, 1/4);
    [train_recs, valid_recs] = get_balanced_fold(SEMAINE_dir, train_recs, au, 1/4);
    
    % load the training and testing data for the current fold
    [~, ~, test_samples, test_labels, raw_test] = Prepare_HOG_AU_data_generic(train_recs, test_recs, au, rest_aus, SEMAINE_dir, hog_data_dir, pca_loc);
    
    [train_samples, train_labels, valid_samples, valid_labels, ~, PC, means, scaling] = Prepare_HOG_AU_data_generic(train_recs, valid_recs, au, rest_aus, SEMAINE_dir, hog_data_dir, pca_loc);
    train_samples = sparse(train_samples);
    valid_samples = sparse(valid_samples);
    test_samples = sparse(test_samples);

    %% Cross-validate here                
    [ best_params, ~ ] = validate_grid_search(svm_train, svm_test, false, train_samples, train_labels, valid_samples, valid_labels, hyperparams);

    model = svm_train(train_labels, train_samples, best_params);        

    [prediction, a, actual_vals] = predict(test_labels, test_samples, model);

    % Go from raw data to the prediction
    w = model.w(1:end-1)';
    b = model.w(end);

    svs = bsxfun(@times, PC, 1./scaling') * w;

    % Attempt own prediction
    preds_mine = bsxfun(@plus, raw_test, -means) * svs + b;

    assert(norm(preds_mine - actual_vals) < 1e-8);

%     name = sprintf('results_ideal/AU_%d_static.dat', au);
% 
%     pos_lbl = model.Label(1);
%     neg_lbl = model.Label(2);
%         
%     write_lin_svm(name, means, svs, b, pos_lbl, neg_lbl);

    name = sprintf('results_ideal/AU_%d_static.mat', au);

    tp = sum(test_labels == 1 & prediction == 1);
    fp = sum(test_labels == 0 & prediction == 1);
    fn = sum(test_labels == 1 & prediction == 0);
    tn = sum(test_labels == 0 & prediction == 0);

    precision = tp/(tp+fp);
    recall = tp/(tp+fn);

    f1 = 2 * precision * recall / (precision + recall);    
    
    save(name, 'model', 'f1', 'precision', 'recall', 'best_params');
        
end

end


