function [result, prediction] = svr_test_linear_shift_learn_local(test_labels, test_samples, model)
   
    prediction = predict(test_labels, test_samples, model);
    
    if(model.cutoff >= 0)
        % perform shifting here per person
        users = unique(model.vid_ids);

        for i=1:numel(users)

            preds_user = prediction(strcmp(model.vid_ids, users(i)));
            prediction_curr =  prediction(strcmp(model.vid_ids, users(i)));
            
            for f=1:numel(preds_user)
                                
                frame_beg = f - model.time_window/2;
                frame_end = f + model.time_window/2;
                
                frame_beg(frame_beg > numel(preds_user)) = numel(preds_user);
                frame_beg(frame_beg < 1) = 1;
                
                frame_end(frame_end > numel(preds_user)) = numel(preds_user);
                frame_end(frame_end < 1) = 1;
                
                sorted = sort(preds_user(frame_beg:frame_end));
            
                % alternative, move to histograms and pick the highest one

                shift = sorted(round(end*model.cutoff)+1);

                prediction_curr(f) = prediction_curr(f) - shift;
            end
            
            prediction(strcmp(model.vid_ids, users(i))) = prediction_curr;

        end
    end
    
    % Cap the prediction as well
    prediction(prediction<0)=0;
    prediction(prediction>5)=5;
    
    % using the average of RMS errors
%     result = mean(sqrt(mean((prediction - test_labels).^2)));  
    result = corr(test_labels, prediction);
    
    if(isnan(result))
        result = 0;
    end
    
end