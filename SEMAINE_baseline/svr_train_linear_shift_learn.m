function [model] = svr_train_linear_shift_learn(train_labels, train_samples, hyper)
    % Change to your downloaded location
    addpath('C:\liblinear\matlab')
    
    comm = sprintf('-s 11 -B 1 -p %.10f -c %.10f -q', hyper.p, hyper.c);
    model = train(train_labels, train_samples, comm);
    
    % Try predicting on the valid samples data and shifting it
        
    cutoffs = 0:0.05:0.8;
    results = zeros(numel(cutoffs)+1, 1);

    % TODO double check if this is correct
    prediction = hyper.valid_samples * model.w(1:end-1)' + model.w(end);

    for c=1:numel(cutoffs)
        % perform shifting here per person
        users = unique(hyper.vid_ids);
        
        prediction_curr = prediction;
        
        for i=1:numel(users)

            preds_user = prediction_curr(strcmp(hyper.vid_ids, users(i)));
            sorted = sort(preds_user);

            % alternative, move to histograms and pick the highest one

            shift = sorted(round(end*cutoffs(c))+1);

            prediction_curr(strcmp(hyper.vid_ids, users(i))) = preds_user - shift;

        end
        
        prediction_curr(prediction_curr<0)=0;
        prediction_curr(prediction_curr>5)=5;
    
        result = corr(hyper.valid_labels, prediction_curr);
        results(c) = result;
    end
    
    % option of no cutoff as well    
    cutoffs = cat(2,cutoffs, -1);
    prediction(prediction<0)=0;
    prediction(prediction>5)=5;
    
    results(end) = corr(hyper.valid_labels, prediction);    

    [best, best_id] = max(results);
    result = results(best_id);
    model.cutoff = cutoffs(best_id);
    model.vid_ids = hyper.vid_ids;
end