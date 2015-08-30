function [result, prediction] = svr_test_linear(test_labels, test_samples, model)

    prediction = test_samples * model.w(1:end-1)' + model.w(end);
%     prediction = predict(test_labels, test_samples, model);
    prediction(~model.success) = 0;
    % All the models should include shifting
    
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