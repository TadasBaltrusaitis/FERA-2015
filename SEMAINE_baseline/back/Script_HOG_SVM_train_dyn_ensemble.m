function Script_HOG_SVM_train_dyn_ensemble()

% Change to your downloaded location
addpath('C:\liblinear\matlab')

%% load shared definitions and AU data
shared_defs;

% Set up the hyperparameters to be validated
hyperparams.c = 10.^(-7:0.5:-2);
%hyperparams.e = 10.^(-6:1:-1);
hyperparams.e = 10.^(-3);

%hyperparams.under_ratio = [1, 1.5, 2, 3, 4, 6];

hyperparams.validate_params = {'c', 'e'};

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
    [ best_params, ~ ] = validate_grid_search_no_par(svm_train, svm_test, false, train_samples, train_labels, valid_samples, valid_labels, hyperparams);

    model = svm_train(train_labels, train_samples, best_params);        
    
    [res, prediction] = svm_test_linear(valid_labels, valid_samples, model);

    % Go from raw data to the prediction
%     w = model.w(1:end-1)';
%     b = model.w(end);
% 
%     svs = bsxfun(@times, PC, 1./scaling') * w;
% 
%     % Attempt own prediction
%     preds_mine = bsxfun(@plus, raw_valid, -means) * svs + b;
% 
%     assert(norm(preds_mine - actual_vals) < 1e-8);

%     name = sprintf('paper_res/AU_%d_dynamic.dat', au);
%         
%     pos_lbl = model.Label(1);
%     neg_lbl = model.Label(2);
%         
%     write_lin_dyn_svm(name, means, svs, b, pos_lbl, neg_lbl);

    name = sprintf('trained_sampling/AU_%d_dynamic_ensemble.mat', au);

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
    comm = sprintf('-s 1 -B 1 -e %.10f -c %.10f -q', hyper.e, hyper.c);
    
    pos_count = sum(train_labels == 1);
    neg_count = sum(train_labels == 0);
    
    max_ensembles = 21;
    
    num_ensembles = min(floor(neg_count /  pos_count), max_ensembles);
    
    if(mod(num_ensembles, 2) == 0)
        num_ensembles = num_ensembles - 1;
    end
    
    if(num_ensembles > 1)
        
        neg_ensemble_size = pos_count;
        
        neg_samples_ids = randperm(neg_count);
        
        model = cell(num_ensembles, 1);
               
        pos_samples = train_samples(train_labels == 1,:);
        pos_labels = train_labels(train_labels == 1,:);                
                 
        neg_samples = train_samples(train_labels == 0,:);
        neg_labels = train_labels(train_labels == 0,:);

        %% for each cluster train a model
        for k=1:num_ensembles

            %% balance the data here:
            neg_samples_ids_c = neg_samples_ids((k-1)*neg_ensemble_size+1:k*neg_ensemble_size);
            neg_samples_k = neg_samples(neg_samples_ids_c, :);
            neg_labels_k = neg_labels(neg_samples_ids_c, :);
                        
            %%
            train_samples_k = cat(1, pos_samples, neg_samples_k);
            train_labels_k = cat(1, pos_labels, neg_labels_k);
            
            model{k} = train(train_labels_k, train_samples_k, comm);
        end
        
    else        
        model = {train(train_labels, train_samples, comm)};
    end
end

function [result, prediction] = svm_test_linear(test_labels, test_samples, model)

    prediction_full = zeros(size(test_labels,1), numel(model));
    
    %%
    for i = 1:numel(model)
        w = model{i}.w(1:end-1)';
        b = model{i}.w(end);

        % Attempt own prediction
        prediction_c = test_samples * w + b;
        l1_inds = prediction_c > 0;
        l2_inds = prediction_c <= 0;
        prediction_c(l1_inds) = model{i}.Label(1);
        prediction_c(l2_inds) = model{i}.Label(2);
        
        prediction_full(:,i) = prediction_c;
        
    end
 
    %%
    % if half or more votes given take this prediciton
    prediction = mean(prediction_full, 2) > 0.5;
    
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
