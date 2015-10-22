function Script_HOG_SVM_train_semaine_static_v_bp4d()

% Change to your downloaded location
addpath('C:\liblinear\matlab')
addpath('../data extraction/');
%% load shared definitions and AU data
shared_defs;

semaine_au = intersect([2,12,17,25,28,45], all_aus);
all_aus_semaine = [2,12,17,25,28,45];

% Set up the hyperparameters to be validated
hyperparams.c = 10.^(-7:1:-1);
hyperparams.e = 10.^(-3);

hyperparams.validate_params = {'c', 'e'};

% Set the training function
svm_train = @svm_train_linear;
    
% Set the test function (the first output will be used for validation)
svm_test = @svm_test_linear;

pca_loc = '../pca_generation/generic_face_rigid.mat';

%%
for a=1:numel(aus)
    
    if(ismember(aus(a), semaine_au))
    
        au = aus(a);
     

        find_BP4D;
        % load the training and testing data for the current fold
        [~, ~, test_samples, test_labels, raw_valid, PC, means, scaling] = Prepare_HOG_AU_data_generic(train_recs, devel_recs, au, BP4D_dir, hog_data_dir, pca_loc);

        valid_samples = test_samples;
        valid_labels = test_labels;
        
        od = cd('../SEMAINE_baseline/');
        find_SEMAINE;
        rest_aus = setdiff(all_aus_semaine, au);        
        [train_samples_semaine, train_labels_semaine, ~, ~, ~, ~, ~, ~] = Prepare_HOG_AU_data_generic(train_recs, devel_recs, au, rest_aus, SEMAINE_dir, hog_data_dir, pca_loc);
        cd(od);

        train_samples = train_samples_semaine;
        train_labels = train_labels_semaine;            
        
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
        preds_mine = bsxfun(@plus, raw_valid, -means) * svs + b;

        assert(norm(preds_mine - actual_vals) < 1e-8);

        name = sprintf('camera_ready/AU_%d_static_SEMAINE_v_bp4d.dat', au);
        
        pos_lbl = model.Label(1);
        neg_lbl = model.Label(2);
        
        write_lin_dyn_svm(name, means, svs, b, pos_lbl, neg_lbl);

        name = sprintf('camera_ready/AU_%d_static_SEMAINE_v_bp4d.mat', au);

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

end

function [model] = svm_train_linear(train_labels, train_samples, hyper)
    comm = sprintf('-s 1 -B 1 -e %.10f -c %.10f -q', hyper.e, hyper.c);
    model = train(train_labels, train_samples, comm);
end

function [result, prediction] = svm_test_linear(test_labels, test_samples, model)

    w = model.w(1:end-1)';
    b = model.w(end);

    % Attempt own prediction
    prediction = test_samples * w + b;
    l1_inds = prediction > 0;
    l2_inds = prediction <= 0;
    prediction(l1_inds) = model.Label(1);
    prediction(l2_inds) = model.Label(2);
 
    %[prediction, a, actual_vals] = predict(test_labels, test_samples, model);           
    
    tp = sum(test_labels == 1 & prediction == 1);
    fp = sum(test_labels == 0 & prediction == 1);
    fn = sum(test_labels == 1 & prediction == 0);
    tn = sum(test_labels == 0 & prediction == 0);

    precision = tp/(tp+fp);
    recall = tp/(tp+fn);

    f1 = 2 * precision * recall / (precision + recall);

%     result = corr(test_labels, prediction);
    fprintf('F1:%.3f\n', f1);
    if(isnan(f1))
        f1 = 0;
    end
    result = f1;
end
