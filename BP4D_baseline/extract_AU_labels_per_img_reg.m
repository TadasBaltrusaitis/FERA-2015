clear
shared_defs
addpath(genpath('../data extraction/'));

out_dir = 'F:/face_datasets/BP4D/labels/';

users = cat(2, train_recs, devel_recs);

for i =1:numel(users)
   
    out_file = fopen(sprintf('%s/%s_reg.txt', out_dir, users{i}), 'w');
    fprintf(out_file, 'filename, success');
    for au=1:numel(all_aus_int)
        fprintf(out_file, ', AU%d', all_aus_int(au));
    end 
    
    [ labels_all, valid_ids, vid_ids  ] = extract_BP4D_labels_intensity(BP4D_dir_int, users(i), all_aus_int);    
    
    fprintf(out_file, '\n');
    
    % Need to grab the filenames 
    dirs = dir(sprintf('F:/face_datasets/BP4D/processed_data/%s*', users{i}));
    
    % Grab successess
    [~, tracked_inds_hog, ~] = Read_HOG_files(users(i), hog_data_dir);
    ind = 1;
    for d=1:numel(dirs)
        
        files = dir(sprintf('F:/face_datasets/BP4D/processed_data/%s/*.png', dirs(d).name));
        
        for f=1:numel(files)
            success = tracked_inds_hog(ind);
            fprintf(out_file, '../processed_data/%s/%s, %d', dirs(d).name, files(f).name, success);

            for au=1:numel(all_aus_int)
                fprintf(out_file, ', %d', labels_all{d}(f,au));
            end
            fprintf(out_file, '\n');
            ind = ind + 1;
        end
    end
    fclose(out_file);
end