clear;

%%
hog_dir = 'C:\tadas\face_datasets\fera_2015\semaine\train\processed\';
hog_files = dir([hog_dir, '*.hog']);

[appearance_data, valid_ids, vid_ids_train] = Read_HOG_files_small(hog_files, hog_dir, 800);
appearance_data = appearance_data(valid_ids, :);
vid_ids_train = vid_ids_train(valid_ids,:);

%%
means_norm = mean(appearance_data);
stds_norm = std(appearance_data);

normed_data = bsxfun(@times, bsxfun(@plus, appearance_data, -means_norm), 1./stds_norm);

[PC, score, eigen_vals] = princomp(normed_data, 'econ');

% Keep 95 percent of variability
total_sum = sum(eigen_vals);
count = numel(eigen_vals);
for i=1:numel(eigen_vals)
   if ((sum(eigen_vals(1:i)) / total_sum) >= 0.95)
      count = i;
      break;
   end
end

PC = PC(:,1:count);

save('semaine_model.mat', 'PC', 'means_norm', 'stds_norm');    
