addpath('../data extraction');
find_BP4D;
pca_loc = '../pca_generation/generic_face_rigid.mat';
od = cd('../BP4D_baseline/');
shared_defs;
% cd(od);
i = 1;
% go through all of them and load the model
[~, ~, valid_samples_bp4d, valid_labels_bp4d, raw_valid, PC, means, scaling] = Prepare_HOG_AU_data_generic(train_recs, devel_recs, all_aus, BP4D_dir, hog_data_dir, pca_loc);
cd(od);
%%
i = 1;
for a=all_aus
   

    load(sprintf('../BP4D_baseline/camera_ready/AU_%d_static_intensity.mat', a));
    
    test_labels = valid_labels_bp4d(:, i);
    
    w = model.w(1:end-1)';
    b = model.w(end);

    svs = bsxfun(@times, PC, 1./scaling') * w;

    % Attempt own prediction
    preds_mine = bsxfun(@plus, raw_valid, -means) * svs + b;
    %%
    preds_mine = smooth(preds_mine, 11);
%     window = 1;
%     preds_mine_extend = zeros(size(preds_mine,1), window);
%     for k=1:window
%         if(k < window/2)
%             sub_preds_mine = preds_mine(k:end-(window-1)/2);
%             preds_mine_extend(1:size(sub_preds_mine,1),k) = sub_preds_mine;
%         elseif(k - 1 > window/2)
%             sub_preds_mine = preds_mine(window-(window-1)/2:end);
%             preds_mine_extend(end-size(sub_preds_mine,1)+1:end,k) = sub_preds_mine;
%         else
%            preds_mine_extend(:,k) = preds_mine; 
%         end
%         
%     end
%     preds_mine = mean(preds_mine_extend,2);
%     preds_mine(preds_mine >= 0.5) = 1;
%     preds_mine(preds_mine < 0.5) = 0;
    %%
    l1_inds = preds_mine > 0;
    l2_inds = preds_mine <= 0;
    preds_mine(l1_inds) = model.Label(1);
    preds_mine(l2_inds) = model.Label(2);
 
    %[prediction, a, actual_vals] = predict(test_labels, test_samples, model);           
    
    tp = sum(test_labels == 1 & preds_mine == 1);
    fp = sum(test_labels == 0 & preds_mine == 1);
    fn = sum(test_labels == 1 & preds_mine == 0);
    tn = sum(test_labels == 0 & preds_mine == 0);

    precision = tp/(tp+fp);
    recall = tp/(tp+fn);

    f1 = 2 * precision * recall / (precision + recall);

%     result = corr(test_labels, prediction);
    fprintf('F1:%.3f\n', f1);
    if(isnan(f1))
        f1 = 0;
    end
    result = f1;
    
    i = i + 1;
end