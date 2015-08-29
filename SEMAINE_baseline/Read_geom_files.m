function [geom_data, valid_ids] = Read_geom_files(users, vid_ids, hog_data_dir)

    geom_data = [];
    valid_ids = [];
    
    load('../pca_generation/pdm_68_aligned_wild.mat');
    
    for i=1:numel(users)
        
        geom_file = [hog_data_dir, '/train/' users{i} '.params.txt'];
        m_file = [hog_data_dir, '/train/' users{i} '.params.mat'];
        if(~exist(geom_file, 'file'))            
            geom_file = [hog_data_dir, '/devel/' users{i} '.params.txt'];
            m_file = [hog_data_dir, '/devel/' users{i} '.params.mat'];
        end
        
        
        if(~exist(m_file, 'file'))
            res = dlmread(geom_file, ' ');
            res = res(vid_ids(i,1)+1:vid_ids(i,2),[1,2, 3:2:end]);    
            save(m_file, 'res');
        else
            load(m_file);
        end
        
        valid = res(:, 2);
        res = res(:, 9:end);
                  
%         actual_locs = bsxfun(@plus, res * V', M');
        actual_locs = res * V';
        res = cat(2, actual_locs, res);

        valid_ids = cat(1, valid_ids, valid);
            
        geom_data = cat(1, geom_data, res);
                
    end
end