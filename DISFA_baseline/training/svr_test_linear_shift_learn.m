function [result, prediction] = svr_test_linear_shift_learn(test_labels, test_samples, model)
   
    fprintf('Predict\n');
    prediction = predict(test_labels, test_samples, model);
    prediction(~model.success) = 0;
    
    if(model.cutoff >= 0)
        % perform shifting here per person
        users = unique(model.vid_ids);

        for i=1:numel(users)

            preds_user = prediction(strcmp(model.vid_ids, users(i)));
            sorted = sort(preds_user);

            % alternative, move to histograms and pick the highest one

            shift = sorted(round(end*model.cutoff)+1);

            prediction(strcmp(model.vid_ids, users(i))) = preds_user - shift;

        end
    end
    
    % Cap the prediction as well
    prediction(prediction<0)=0;
    prediction(prediction>5)=5;
    
    % using the average of RMS errors
%     result = mean(sqrt(mean((prediction - test_labels).^2)));  
    result = corr(test_labels, prediction);
    [ ~, ~, ~, ccc, ~, ~ ] = evaluate_classification_results( prediction, test_labels ); 
    
    result = ccc;
    
    if(isnan(result))
        result = 0;
    end
    
end