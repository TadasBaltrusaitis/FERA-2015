function [result, prediction] = svm_test_linear_shift(test_labels, test_samples, model)

    w = model.w(1:end-1)';
    b = model.w(end);

    % Attempt own prediction
    prediction = test_samples * w + b;

    if(model.cutoff >= 0)
        % perform shifting here per person
        users = unique(model.vid_ids);

        for i=1:numel(users)

            preds_user = prediction(strcmp(model.vid_ids, users(i)));

            if(model.Label(1) == 1)
                sorted = sort(preds_user, 'descend');
                shift = sorted(round(end*model.cutoff)+1);
            else
                sorted = sort(preds_user, 'ascend');
                shift = sorted(end - round(end*model.cutoff));
            end

            % alternative, move to histograms and pick the highest one
            prediction(strcmp(model.vid_ids, users(i))) = preds_user - shift;
        end
    end

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