function [file_ids, labels, good_inds] = extract_ck_files(ck_dir, au)

    file_ids = cell(593, 1);
    labels = zeros(593, 1);
    
    label_loc = [ck_dir, '/FACS_labels/FACS/'];
    
    subjects = dir(label_loc);
    
    f_so_far = 0;
    good_inds = true(593, 1);
    
    for i=3:numel(subjects)
       
        recordings = dir([label_loc, subjects(i).name]);

        for j=3:numel(recordings)
        
            f_so_far = f_so_far + 1;
            
            file_ids{f_so_far} = [subjects(i).name '_' recordings(j).name];
            
            file = dir([label_loc, subjects(i).name, '/' recordings(j).name '/*.txt']);
            
            aus = dlmread([label_loc, subjects(i).name, '/' recordings(j).name '/' file.name]);
                        
            label = aus(aus(:,1) == au, 2);
            
            if(~isempty(label))
                if(label == 0)
                    label = 3;
                    good_inds(f_so_far) = 0;
                end
                labels(f_so_far) = label;
            end
        end
            
    end

end