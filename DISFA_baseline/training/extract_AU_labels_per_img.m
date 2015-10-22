clear
shared_defs

out_dir = 'F:/face_datasets/DISFA/labels/';

for i =1:numel(users)
   
    out_file = fopen(sprintf('%s/%s.txt', out_dir, users{i}), 'w');
    fprintf(out_file, 'filename, success');
    labels_all = [];
    for au=1:numel(all_aus)
        fprintf(out_file, ', AU%d', all_aus(au));
        labels = extract_au_labels({['D:\Datasets\DISFA\ActionUnit_Labels/' users{i} '/' users{i}]}, all_aus(au));
        labels_all = cat(2, labels_all, labels);
    end 
    fprintf(out_file, '\n');
    
    % Need to grab the filenames 
    files = dir(sprintf('F:/face_datasets/DISFA/processed_data/LeftVideo%s_comp/*.png', users{i}));
    
    % Grab successess
    [~, tracked_inds_hog, ~] = Read_HOG_files(users(i), hog_data_dir);
    
    for f=1:size(labels_all,1)
       
        fprintf(out_file, '../processed_data/LeftVideo%s_comp/%s, %d', users{i}, files(f).name, tracked_inds_hog(i));
        
        for au=1:numel(all_aus)
            fprintf(out_file, ', %d', labels_all(f,au));
        end
        fprintf(out_file, '\n');
    end
    
    files = dir(sprintf('F:/face_datasets/DISFA/processed_data/RightVideo%s_comp/*.png', users{i}));
    
    % Now do the same for right
    for f=1:size(labels_all,1)
       
        fprintf(out_file, '../processed_data/RightVideo%s_comp/%s, %d', users{i}, files(f).name, tracked_inds_hog(i));
        
        for au=1:numel(all_aus)
            fprintf(out_file, ', %d', labels_all(f,au));
        end
        fprintf(out_file, '\n');
    end
    
    fclose(out_file);
end