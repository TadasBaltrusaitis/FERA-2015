addpath('../data extraction');
find_BP4D;
pca_loc = '../pca_generation/generic_face_rigid.mat';
od = cd('../BP4D_baseline/');
shared_defs;
% cd(od);
i = 1;
% go through all of them and load the model
[~, ~, ~, valid_samples_bp4d, valid_labels_bp4d, vid_ids_devel_string, raw_valid, PC, means, scaling] = Prepare_HOG_AU_data_generic_intensity(train_recs, devel_recs, all_aus_int, BP4D_dir_int, hog_data_dir, pca_loc);
cd(od);
all_users = unique(vid_ids_devel_string);

%%
i = 1;
fprintf('-------------------\n');
for a=all_aus_int
   
    %%
    load(sprintf('../BP4D_baseline/camera_ready/AU_%d_static_intensity.mat', a));
    
    test_labels = valid_labels_bp4d(:, i);
    
    w = model.w(1:end-1)';
    b = model.w(end);

    svs = bsxfun(@times, PC, 1./scaling') * w;

    % Attempt own prediction
    preds_mine = bsxfun(@plus, raw_valid, -means) * svs + b;
    preds_mine = smooth(preds_mine, 7);
    
%     fprintf('%d\n', sum(preds_mine<0));
%     fprintf('%d\n', sum(preds_mine>5));
%     hist(preds_mine);

    %%
    
    % get rid of outliers first
    preds_mine(preds_mine < -0.75) = -0.75;
    
    preds_mine_s = preds_mine;
    preds_mine_s(preds_mine_s<=0) = 0;
    preds_mine_s(preds_mine_s>5) = 5;
    
    for u=all_users'
       user_ids_all = strcmp(vid_ids_devel_string, u);
       inds = 1:numel(user_ids_all);
       inds = inds(user_ids_all);
       steps = floor(linspace(1, numel(inds), 9));
       
       mse_orig = mean((preds_mine_s(user_ids_all) - test_labels(user_ids_all)).^2);
       
       f = figure;
       plot(preds_mine(user_ids_all), 'r');
       for m=1:8
           user_ids = inds(steps(m):steps(m+1));
           vals = preds_mine(user_ids);

           v_sort = sort(vals);
           cutoff = v_sort(round(end/5));
           if(cutoff < 0)
              preds_mine(user_ids) = preds_mine(user_ids) - cutoff;
           end

       end
       %%
       preds_mine(user_ids_all & preds_mine<=0) = 0;
       preds_mine(user_ids_all & preds_mine>5) = 5;
       
       mse_new = mean(preds_mine(user_ids_all) - test_labels(user_ids_all));
       
       
       hold on;
       plot(preds_mine(user_ids_all), 'b');
       plot(test_labels(user_ids_all), 'g');
       if(mse_new > mse_orig)
%            fprintf('user:%s, au%d, mse_old %.3f, mse_new %.3f \n', u{1}, a, mse_orig, mse_new);
        pause
       end
       close(f);
    end
    
    
    correlation = corr(test_labels, preds_mine);
    MSE = mean((test_labels - preds_mine).^2);
    fprintf('Corr:%.3f, mse:%.3f\n', correlation, MSE);

    i = i + 1;
end