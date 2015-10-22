function [model] = svm_train_linear_shift(train_labels, train_samples, hyper)

    addpath('C:\liblinear\matlab')
    comm = sprintf('-s 1 -B 1 -e %.10f -c %.10f -q', hyper.e, hyper.c);
    model = train(train_labels, train_samples, comm);
   
    % Try predicting on the valid samples data and shifting it     
    w = model.w(1:end-1)';
    b = model.w(end);

    % Attempt own prediction
    prediction = hyper.valid_samples * w + b;
        
    cutoffs = 0:0.025:0.5;
    results = zeros(numel(cutoffs)+1, 1);

    for c=1:numel(cutoffs)
        % perform shifting here per person
        users = unique(hyper.vid_ids);
        
        prediction_curr = prediction;
        
        for i=1:numel(users)

            preds_user = prediction_curr(strcmp(hyper.vid_ids, users(i)));
            if(model.Label(1) == 1)
                sorted = sort(preds_user, 'descend');
            else
                sorted = sort(preds_user, 'ascend');
            end
            % alternative, move to histograms and pick the highest one

            if(model.Label(1) == 1)
                shift = sorted(round(end*cutoffs(c))+1);
                prediction_curr(strcmp(hyper.vid_ids, users(i))) = preds_user - shift;
            else
                shift = sorted(end - round(end*cutoffs(c)));
                prediction_curr(strcmp(hyper.vid_ids, users(i))) = preds_user - shift;
            end

        end

        l1_inds = prediction_curr > 0;
        l2_inds = prediction_curr <= 0;

        prediction_curr(l1_inds) = model.Label(1);
        prediction_curr(l2_inds) = model.Label(2);    
    
        f1 = computeF1(hyper.valid_labels, prediction_curr);

        result = f1;
        results(c) = result;
    end
    
    % option of no cutoff as well    
    cutoffs = cat(2,cutoffs, -1);

    l1_inds = prediction > 0;
    l2_inds = prediction <= 0;

    prediction(l1_inds) = model.Label(1);
    prediction(l2_inds) = model.Label(2);      

    results(end) = computeF1(hyper.valid_labels, prediction);

    [best, best_id] = max(results);
    result = results(best_id);
    model.cutoff = cutoffs(best_id);
    model.vid_ids = hyper.vid_ids;
    
end

function f1 = computeF1(test_labels, prediction)

    tp = sum(test_labels == 1 & prediction == 1);
    fp = sum(test_labels == 0 & prediction == 1);
    fn = sum(test_labels == 1 & prediction == 0);
    tn = sum(test_labels == 0 & prediction == 0);

    precision = tp/(tp+fp);
    recall = tp/(tp+fn);

    f1 = 2 * precision * recall / (precision + recall);

    if(isnan(f1))
        f1 = 0;
    end

end