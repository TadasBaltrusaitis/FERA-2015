function Script_HOG_SVM_train_DISFA_test()

% Change to your downloaded location
addpath('C:\liblinear\matlab')

test_folds = 5;

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
all_recs_SEMAINE = cat(2,train_recs, devel_recs);

aus = [2, 12,17,25];
%%
for a=1:numel(aus)
    
    for t = 1:test_folds
        au = aus(a);

        rest_aus = setdiff(all_aus, au);        

        [train_recs, valid_SEMAINE_recs] = get_balanced_fold(SEMAINE_dir, all_recs_SEMAINE, au, 1/4);

        % load the training and testing data for the current fold
        [train_samples, train_labels, valid_samples, valid_labels, ~, PC, means, scaling] = Prepare_HOG_AU_data_generic(train_recs, valid_SEMAINE_recs, au, rest_aus, SEMAINE_dir, hog_data_dir, pca_loc);

        train_samples = sparse(train_samples);
        valid_samples = sparse(valid_samples);

        %% Cross-validate here                
        [ best_params, ~ ] = validate_grid_search_no_par(svm_train, svm_test, false, train_samples, train_labels, valid_samples, valid_labels, hyperparams);
        model = svm_train(train_labels, train_samples, best_params);        

        % Test on DISFA
        find_DISFA;
        od = cd('../DISFA_baseline/training/');
        all_disfa = [1,2,4,5,6,9,12,15,17,20,25,26];
        rest_aus = setdiff(all_disfa, au);    
        [valid_samples_disfa, valid_labels_disfa, test_samples_disfa, test_labels_disfa] = Prepare_HOG_AU_data_generic(users, au, rest_aus, hog_data_dir);
        cd(od);
        
        % Fine-tune on DISFA
        w = model.w(1:end-1)';
        b = model.w(end);

        % Attempt own prediction
        prediction_DISFA_valid = valid_samples_disfa * w + b;

        train_DISFA = cat(2, ones(numel(prediction_DISFA_valid),1), prediction_DISFA_valid);
        p = train_DISFA / valid_labels_disfa;
        
%         b2 = b + p(1);
%         w2 = w * p(2);
%         prediction_DISFA_valid_2 = valid_samples_disfa * w2 + b2;

        model.w(1:end-1) = model.w(1:end-1) * p(2);
        model.w(end) = model.w(end) + p(1);

        % can now either determin the shift or predict straight

    %     name = sprintf('results_ideal/AU_%d_static.dat', au);
    % 
    %     pos_lbl = model.Label(1);
    %     neg_lbl = model.Label(2);
    %         
    %     write_lin_svm(name, means, svs, b, pos_lbl, neg_lbl);

    end

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


