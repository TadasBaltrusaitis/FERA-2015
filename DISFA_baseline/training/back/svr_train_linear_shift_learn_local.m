function [model] = svr_train_linear_shift_learn_local(train_labels, train_samples, hyper)
    % Change to your downloaded location
    addpath('C:\liblinear\matlab')
    
    comm = sprintf('-s 11 -B 1 -p %.10f -c %.10f -q', hyper.p, hyper.c);
    model = train(train_labels, train_samples, comm);
    
    % Try predicting on the valid samples data and shifting it
        
    cutoffs = 0:0.05:0.8;
    window_sizes = [120:20:240];
    
    results = zeros(numel(window_sizes) * numel(cutoffs) + 1, 1);
    param_vals = struct;
    prediction = hyper.valid_samples * model.w(1:end-1)' + model.w(end);       
    
    ind = 0;
    
    for w=1:numel(window_sizes)
        
        time_window = window_sizes(w) * 20;

        prediction_curr = repmat(prediction, 1, numel(cutoffs));

        % perform shifting here per person
        users = unique(hyper.vid_ids);

        for i=1:numel(users)

            preds_user = prediction_curr(strcmp(hyper.vid_ids, users(i)),:);
            preds_user_full = preds_user(:,1);    
            for f=1:numel(preds_user_full)

                frame_beg = f - time_window/2;
                frame_end = f + time_window/2;

                frame_beg(frame_beg > numel(preds_user_full)) = numel(preds_user_full);
                frame_beg(frame_beg < 1) = 1;

                frame_end(frame_end > numel(preds_user_full)) = numel(preds_user_full);
                frame_end(frame_end < 1) = 1;

                sorted = sort(preds_user_full(frame_beg:frame_end));

                % alternative, move to histograms and pick the highest one
                for c=1:numel(cutoffs)  
                    shift = sorted(round(end*cutoffs(c))+1);
                    preds_user(f,c) = preds_user_full(f) - shift;
                end
            end
            prediction_curr(strcmp(hyper.vid_ids, users(i)),:) = preds_user;

        end

        prediction_curr(prediction_curr<0)=0;
        prediction_curr(prediction_curr>5)=5;

        for c=1:numel(cutoffs)            
            ind = ind+1;
            param_vals(ind).time_window = time_window;
            param_vals(ind).cutoff = cutoffs(c);
            result = corr(hyper.valid_labels, prediction_curr(:,c));            
            results(ind) = result;
        end
    end
    
    % option of no cutoff as well    
    prediction(prediction<0)=0;
    prediction(prediction>5)=5;
    
    results(end) = corr(hyper.valid_labels, prediction);    
    param_vals(ind+1).cutoff = -1;
    param_vals(ind+1).time_window = 0;
    
    [best, best_id] = max(results);
    result = results(best_id);
    model.cutoff = param_vals(best_id).cutoff;
    model.vid_ids = hyper.vid_ids;
    model.time_window = param_vals(best_id).time_window;
    
end