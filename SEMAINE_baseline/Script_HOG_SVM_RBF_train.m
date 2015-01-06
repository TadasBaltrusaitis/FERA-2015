function Script_HOG_SVM_RBF_train()

% Change to your downloaded location
addpath('C:\libsvm\matlab')

%% load shared definitions and AU data
shared_defs;

% Set up the hyperparameters to be validated
hyperparams.c = 10.^(-6:2:1);
hyperparams.g = 10.^(-6:2:-3);

hyperparams.validate_params = {'c', 'g'};

% Set the training function
svm_train = @svm_train_rbf;
    
% Set the test function (the first output will be used for validation)
svm_test = @svm_test_rbf;

pca_loc = '../pca_generation/generic_face_rigid.mat';

%%
for a=1:numel(aus)
    
    au = aus(a);
            
    rest_aus = setdiff(all_aus, au);        

    % load the training and testing data for the current fold
    [train_samples, train_labels, valid_samples, valid_labels, raw_valid, PC, means, scaling] = Prepare_HOG_AU_data_generic(train_recs, devel_recs, au, rest_aus, SEMAINE_dir, hog_data_dir, pca_loc);
    
    train_labels(train_labels == 0) = -1;
    valid_labels(valid_labels == 0) = -1;
    
    train_samples = train_samples(1:2:end,:);
    train_labels = train_labels(1:2:end,:);

    valid_samples = valid_samples(1:4:end,:);
    valid_labels = valid_labels(1:4:end,:);
    
    %% Cross-validate here                
    [ best_params, ~ ] = validate_grid_search(svm_train, svm_test, false, train_samples, train_labels, valid_samples, valid_labels, hyperparams);

    best_params

    model = svm_train(train_labels, train_samples, best_params);        

    [prediction, a, actual_vals] = svmpredict(valid_labels, valid_samples, model);

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
% 
%     name = sprintf('trained/AU_%d_RBF_static.dat', au);
% 
%     write_lin_svr(name, means, svs, b);

    name = sprintf('trained/AU_%d_RBF_static.mat', au);

    tp = sum(valid_labels == 1 & prediction == 1);
    fp = sum(valid_labels == -1 & prediction == 1);
    fn = sum(valid_labels == 1 & prediction == -1);
    tn = sum(valid_labels == -1 & prediction == -1);

    precision = tp/(tp+fp);
    recall = tp/(tp+fn);

    f1 = 2 * precision * recall / (precision + recall);    
    
    save(name, 'model', 'f1', 'precision', 'recall');
        
end

end

function [model] = svm_train_rbf(train_labels, train_samples, hyper)

    comm = sprintf('-s 0 -t 2 -c %f -g %f -q', hyper.c, hyper.g);
    
    model = svmtrain(train_labels, train_samples, comm);
end

function [result, prediction] = svm_test_rbf(test_labels, test_samples, model)

    [prediction, a, actual_vals] = svmpredict(test_labels, test_samples, model);           
    
    tp = sum(test_labels == 1 & prediction == 1);
    fp = sum(test_labels == -1 & prediction == 1);
    fn = sum(test_labels == 1 & prediction == -1);
    tn = sum(test_labels == -1 & prediction == -1);

    precision = tp/(tp+fp);
    recall = tp/(tp+fn);

    f1 = 2 * precision * recall / (precision + recall);

%     result = corr(test_labels, prediction);
    
    if(isnan(f1))
        f1 = 0;
    end
    result = f1;
    
    fprintf('F1:%.3f\n', f1);
end
