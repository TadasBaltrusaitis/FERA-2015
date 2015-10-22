% Change to your downloaded location
clear
addpath('C:\liblinear\matlab')
addpath('../data extraction/');
%% load shared definitions and AU data

% Set up the hyperparameters to be validated
hyperparams_reg.c = 10.^(-7:1:5);
hyperparams_reg.p = 10.^(-2);

hyperparams_reg.validate_params = {'c', 'p'};

hyperparams_class.c = 10.^(-8:1:3);
hyperparams_class.e = 10.^(-3);

hyperparams_class.validate_params = {'c', 'e'};


% Set the training function
svm_train = @svm_train_linear;
    
% Set the test function (the first output will be used for validation)
svm_test = @svm_test_linear;

svr_train = @svr_train_linear;
svr_test = @svr_test_linear;

pca_loc = '../pca_generation/generic_face_rigid.mat';

aus = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26];

%%
for a=1:numel(aus)
            
    au_current = aus(a);

    rest_aus = setdiff(aus, au_current);        

    % First grab a DISFA test subset
    % create the training and validation data
    find_DISFA;
    od = cd('../DISFA_baseline/training/');
    users_all_DISFA = users;        
    
    % make sure validation data's labels are balanced
    [users_train_disfa, users_test_DISFA] = get_balanced_fold(DISFA_dir, users_all_DISFA, au_current, 1/5, 1);        
    [users_train_disfa, users_valid_DISFA] = get_balanced_fold(DISFA_dir, users_train_disfa, au_current, 1/4, 1);
    
    hog_data_dir_DISFA = hog_data_dir;
    
    % Grab disfa test, train and valid samples    
    [~, ~, test_samples, test_labels, ~, ~, ~, ~, test_ids, test_success] = ...
        Prepare_HOG_AU_data_generic({}, users_test_DISFA, au_current, rest_aus, hog_data_dir_DISFA);

    % need to split the rest
    [train_samples_d, train_labels_d, valid_samples_d, valid_labels_d, ~, ~, ~, ~, valid_ids_d, valid_success_d] = ...
        Prepare_HOG_AU_data_generic(users_train_disfa, users_valid_DISFA, au_current, rest_aus, hog_data_dir_DISFA);    
    
    train_samples_d = sparse(train_samples_d);    
    
    cd(od);
        
    use_semaine = ~isempty(intersect(au_current, [2, 12, 17, 25]));
    use_disfa = true;
    use_bp4d_class = ~isempty(intersect(au_current, [1, 2, 4, 15]));
    use_bp4d_int = ~isempty(intersect(au_current, [6, 12, 17]));
    use_unbc = ~isempty(intersect(au_current, [4, 6, 7, 9, 10, 12, 20, 25, 26]));
    
    %%
    if(use_semaine)
        % These have been trained already so just load them
        name = sprintf('../SEMAINE_baseline/results_regression/AU_%d_static_shift.mat', au_current);
        load(name, 'model');
        model_sem = model;
        % Convert to a regressor from the classifier
        prediction_DISFA_train = train_samples_d * model_sem.w(1:end-1)' + model_sem.w(end);
        prediction_DISFA_valid = valid_samples_d * model_sem.w(1:end-1)' + model_sem.w(end);

        train_DISFA = cat(2, ones(numel(prediction_DISFA_train),1), prediction_DISFA_train);
        
        % learn a mapping from the classifier to a regressor
        hyperparams_reg.success = valid_success_d;
        hyperparams_reg.valid_samples = prediction_DISFA_valid;
        hyperparams_reg.valid_labels = valid_labels_d;
        hyperparams_reg.vid_ids = valid_ids_d;    
        prediction_DISFA_train = sparse(prediction_DISFA_train);
        [ best_params, ~ ] = validate_grid_search_no_par(svr_train, svr_test, false, prediction_DISFA_train, train_labels_d, prediction_DISFA_valid, valid_labels_d, hyperparams_reg);
        model_reg = svr_train(train_labels_d, prediction_DISFA_train, best_params);             
        
        p = train_DISFA \ train_labels_d;
        
        p = model_reg.w;
        
        model_sem.w(1:end) = model_sem.w(1:end) * p(1);
        model_sem.w(end) = model_sem.w(end) + p(2);   

    end
    
    %%
    if(use_bp4d_class)
        % These have been trained already so just load them
        name = sprintf('../BP4D_baseline/results_regression/AU_%d_static.mat', au_current);
        load(name, 'model');
        model_bp4d = model;
        
        %%
        prediction_DISFA_train = train_samples_d * model_bp4d.w(1:end-1)' + model_bp4d.w(end);
        prediction_DISFA_valid = valid_samples_d * model_bp4d.w(1:end-1)' + model_bp4d.w(end);

        train_DISFA = cat(2, ones(numel(prediction_DISFA_train),1), prediction_DISFA_train);
        
        % learn a mapping from the classifier to a regressor
        hyperparams_reg.success = valid_success_d;
        hyperparams_reg.valid_samples = prediction_DISFA_valid;
        hyperparams_reg.valid_labels = valid_labels_d;
        hyperparams_reg.vid_ids = valid_ids_d;    
        prediction_DISFA_train = sparse(prediction_DISFA_train);
        [ best_params, ~ ] = validate_grid_search_no_par(svr_train, svr_test, false, prediction_DISFA_train, train_labels_d, prediction_DISFA_valid, valid_labels_d, hyperparams_reg);
        model_reg = svr_train(train_labels_d, prediction_DISFA_train, best_params);             
        
        p = train_DISFA \ train_labels_d;
        
        p = model_reg.w;
        
        model_bp4d.w(1:end) = model_bp4d.w(1:end) * p(1);
        model_bp4d.w(end) = model_bp4d.w(end) + p(2);   
    end
    
    if(use_bp4d_int)
       
        % first need to create the models and stuff?
        % These have been trained already so just load them
        name = sprintf('../BP4D_baseline/results_regression/AU_%d_static.mat', au_current);
        load(name, 'model');
        model_bp4d = model;        
                
        %%
        prediction_DISFA_train = train_samples_d * model_bp4d.w(1:end-1)' + model_bp4d.w(end);
        prediction_DISFA_valid = valid_samples_d * model_bp4d.w(1:end-1)' + model_bp4d.w(end);

        train_DISFA = cat(2, ones(numel(prediction_DISFA_train),1), prediction_DISFA_train);
        
        % Learn a mapping to DISFA like regression
        hyperparams_reg.success = valid_success_d;
        hyperparams_reg.valid_samples = prediction_DISFA_valid;
        hyperparams_reg.valid_labels = valid_labels_d;
        hyperparams_reg.vid_ids = valid_ids_d;    
        prediction_DISFA_train = sparse(prediction_DISFA_train);
        [ best_params, ~ ] = validate_grid_search_no_par(svr_train, svr_test, false, prediction_DISFA_train, train_labels_d, prediction_DISFA_valid, valid_labels_d, hyperparams_reg);
        model_reg = svr_train(train_labels_d, prediction_DISFA_train, best_params);             
        
        p = train_DISFA \ train_labels_d;
        
        p = model_reg.w;
        
        model_bp4d.w(1:end) = model_bp4d.w(1:end) * p(1);
        model_bp4d.w(end) = model_bp4d.w(end) + p(2);   
    end
    
    %%
    % Train the DISFA model here
    hyperparams_reg.success = valid_success_d;
    hyperparams_reg.valid_samples = valid_samples_d;
    hyperparams_reg.valid_labels = valid_labels_d;
    hyperparams_reg.vid_ids = valid_ids_d;    
    [ best_params, ~ ] = validate_grid_search_no_par(svr_train, svr_test, false, train_samples_d, train_labels_d, valid_samples_d, valid_labels_d, hyperparams_reg);

    model_d = svr_train(train_labels_d, train_samples_d, best_params);        
    
    % Now need to fuse all of them on validation data
    predictions = [];%ones(numel(valid_labels_d), 1);
    predictions_t = [];%ones(numel(valid_labels_d), 1);
    
    if(use_bp4d_class || use_bp4d_int)        
        prediction_bp4d_valid = valid_samples_d * model_bp4d.w(1:end-1)' + model_bp4d.w(end);
        prediction_bp4d_train = train_samples_d * model_bp4d.w(1:end-1)' + model_bp4d.w(end);
        predictions = cat(2, predictions, prediction_bp4d_valid);
        predictions_t = cat(2, predictions_t, prediction_bp4d_train);
    end
    
    if(use_semaine)
        predictions_sem_valid = valid_samples_d * model_sem.w(1:end-1)' + model_sem.w(end);
        predictions_sem_train = train_samples_d * model_sem.w(1:end-1)' + model_sem.w(end);
        predictions = cat(2, predictions, predictions_sem_valid);
        predictions_t = cat(2, predictions_t, predictions_sem_train);
    end
    
    prediction_d_valid = valid_samples_d * model_d.w(1:end-1)' + model_d.w(end);
    predictions = cat(2, predictions, prediction_d_valid);
    
    prediction_d_train = train_samples_d * model_d.w(1:end-1)' + model_d.w(end);
    predictions_t = cat(2, predictions_t, prediction_d_train);
    
    predictions(~valid_success_d,:) = 0;
    
    hyperparams_reg.success = valid_success_d;    
    predictions_t = sparse(predictions_t);
    [ best_params, ~ ] = validate_grid_search_no_par(svr_train, svr_test, false, predictions_t, train_labels_d, predictions, valid_labels_d, hyperparams_reg);
    model_fuse = svr_train(train_labels_d, predictions_t, best_params);        
    
    % TODO use svr for this instead (with train and validation data?) train
    % on valid, valid on train?
%     p = predictions \ valid_labels_d;
    p = model_fuse.w;
    
    model_full = model_fuse;
    model_full.w = model_d.w * p(end-1);
    
    if(use_semaine)
        model_full.w = model_full.w  + model_sem.w * p(end-2);
    end
    
    if(use_bp4d_class || use_bp4d_int )
       if(~use_semaine)
           model_full.w = model_full.w  + model_bp4d.w * p(end-2);
       else
           model_full.w = model_full.w + model_bp4d.w * p(end-3);
       end
    end
    
    model_full.w(end) = model_full.w(end) + p(end);    
    model_full.success = valid_success_d;
    model_full.valid_samples = predictions;
    model_full.valid_labels = valid_labels_d;
    model_full.vid_ids = valid_ids_d;   
    result = svr_test(valid_labels_d, valid_samples_d, model_full);  
    
    predictions_DISFA_test = test_samples * model_full.w(1:end-1)' + model_full.w(end);
    predictions_DISFA_test(~test_success) = 0;

    predictions_DISFA_test(predictions_DISFA_test < 0) = 0;
    predictions_DISFA_test(predictions_DISFA_test > 5) = 5;

    [ accuracies, F1s, corrs, ccc, rms, classes ] = evaluate_classification_results( predictions_DISFA_test, test_labels );
    
    % so we can create a combined model now
    
    % Now learn about the shifting with all of them on validation
    
    % Finally evaluate on the test partition
            
    name = sprintf('independently/AU_%d_static.mat', au_current);

    save(name, 'model_full', 'F1s', 'corrs', 'ccc', 'rms');

end
