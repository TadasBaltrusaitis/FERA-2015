clear
shared_defs
addpath(genpath('../data extraction/'));

out_dir = 'F:/face_datasets/SEMAINE/labels/';

users = cat(2, train_recs, devel_recs);

for i =1:numel(users)
   
    out_file = fopen(sprintf('%s/%s_class.txt', out_dir, users{i}), 'w');
    fprintf(out_file, 'filename, success');
    for au=1:numel(all_aus)
        fprintf(out_file, ', AU%d', all_aus(au));
    end 
    
    [ labels_all, valid_ids, vid_ids  ] = extract_SEMAINE_labels(SEMAINE_dir, users(i), all_aus);    
    
    fprintf(out_file, '\n');
    
    % Need to grab the filenames 
    files = dir(sprintf('F:/face_datasets/SEMAINE/processed_data/%s/*.png', users{i}));
    
    % Grab successess
    [~, tracked_inds_hog, ~] = Read_HOG_files(users(i), vid_ids, hog_data_dir);
    
    files = files(vid_ids(1):vid_ids(2)-1);
    
    for f=1:numel(files)
        success = tracked_inds_hog(f);
        fprintf(out_file, '../processed_data/%s/%s, %d', users{i}, files(f).name, success);

        for au=1:numel(all_aus)
            fprintf(out_file, ', %d', labels_all{1}(f,au));
        end
        fprintf(out_file, '\n');
    end
        
    fclose(out_file);
end