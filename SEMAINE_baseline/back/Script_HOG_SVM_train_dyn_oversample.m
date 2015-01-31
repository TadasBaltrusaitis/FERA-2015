function Script_HOG_SVM_train_dyn_oversample()

% Change to your downloaded location
addpath('C:\liblinear\matlab')

%% load shared definitions and AU data
shared_defs;

% Set up the hyperparameters to be validated
hyperparams.c = 10.^(-6:1:-4);
%hyperparams.e = 10.^(-6:1:-1);
hyperparams.e = 10.^(-3);

hyperparams.under_ratio = [1];

% How many more samples to generate of positives
hyperparams.over_ratio = [0, 0.25, 0.5, 1, 2, 3, 4];

hyperparams.validate_params = {'c', 'e', 'under_ratio', 'over_ratio'};

% Set the training function
svm_train = @svm_train_linear;
    
% Set the test function (the first output will be used for validation)
svm_test = @svm_test_linear;

pca_loc = '../pca_generation/generic_face_rigid.mat';

%%
for a=1:numel(aus)
    
    au = aus(a);
            
    rest_aus = setdiff(all_aus, au);        

    % load the training and testing data for the current fold
    [train_samples, train_labels, valid_samples, valid_labels, raw_valid, PC, means, scaling] = Prepare_HOG_AU_data_generic_dynamic(train_recs, devel_recs, au, rest_aus, SEMAINE_dir, hog_data_dir, pca_loc);

    train_samples = sparse(train_samples);
    valid_samples = sparse(valid_samples);

    %% Cross-validate here                
    [ best_params, ~ ] = validate_grid_search(svm_train, svm_test, false, train_samples, train_labels, valid_samples, valid_labels, hyperparams);

    model = svm_train(train_labels, train_samples, best_params);        

    [prediction, a, actual_vals] = predict(valid_labels, valid_samples, model);

    % Go from raw data to the prediction
    w = model.w(1:end-1)';
    b = model.w(end);

    svs = bsxfun(@times, PC, 1./scaling') * w;

    % Attempt own prediction
    preds_mine = bsxfun(@plus, raw_valid, -means) * svs + b;

    assert(norm(preds_mine - actual_vals) < 1e-8);

%     name = sprintf('paper_res/AU_%d_dynamic.dat', au);
%         
%     pos_lbl = model.Label(1);
%     neg_lbl = model.Label(2);
%         
%     write_lin_dyn_svm(name, means, svs, b, pos_lbl, neg_lbl);

    name = sprintf('trained_sampling/AU_%d_dynamic_over.mat', au);

    tp = sum(valid_labels == 1 & prediction == 1);
    fp = sum(valid_labels == 0 & prediction == 1);
    fn = sum(valid_labels == 1 & prediction == 0);
    tn = sum(valid_labels == 0 & prediction == 0);

    precision = tp/(tp+fp);
    recall = tp/(tp+fn);

    f1 = 2 * precision * recall / (precision + recall);    
    
    save(name, 'model', 'f1', 'precision', 'recall', 'best_params');
        
end

end

function [model] = svm_train_linear(train_labels, train_samples, hyper)
    comm = sprintf('-s 1 -B 1 -e %f -c %f -q', hyper.e, hyper.c);
    
    pos_count = sum(train_labels == 1);
    neg_count = sum(train_labels == 0);
    
    if(hyper.over_ratio > 0)
       
        inds_train = 1:size(train_labels,1);
        pos_samples = inds_train(train_labels == 1);
        
        % do not produce more pos than neg
        extra_num = min([round(pos_count * hyper.over_ratio), neg_count - pos_count]);
        
        extra_pos = round(linspace(1, size(pos_samples, 2), extra_num));
        
        train_labels_extra = train_labels(extra_pos, :);
        train_samples_extra = train_samples(extra_pos, :);
        
        train_labels = cat(1, train_labels, train_labels_extra);
        train_samples = cat(1, train_samples, train_samples_extra);        
        
    end
    
    pos_count = sum(train_labels == 1);
    neg_count = sum(train_labels == 0);
        
    if(pos_count * hyper.under_ratio < neg_count)
    
        % Remove two thirds of negative examples (to balance the training data a bit)
        inds_train = 1:size(train_labels,1);
        neg_samples = inds_train(train_labels == 0);
        reduced_inds = true(size(train_labels,1),1);
        to_rem = round(neg_count -  pos_count * hyper.under_ratio);
        neg_samples = neg_samples(round(linspace(1, size(neg_samples,2), to_rem)));
        
        reduced_inds(neg_samples) = false;

        train_labels = train_labels(reduced_inds, :);
        train_samples = train_samples(reduced_inds, :);
        
    end
        
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
 
    tp = sum(test_labels == 1 & prediction == 1);
    fp = sum(test_labels == 0 & prediction == 1);
    fn = sum(test_labels == 1 & prediction == 0);
    tn = sum(test_labels == 0 & prediction == 0);

    precision = tp/(tp+fp);
    recall = tp/(tp+fn);

    f1 = 2 * precision * recall / (precision + recall);

    fprintf('F1:%.3f\n', f1);
    if(isnan(f1))
        f1 = 0;
    end
    result = f1;
end
