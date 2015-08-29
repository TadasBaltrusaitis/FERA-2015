function [geom_data, valid_ids] = Read_geom_files(users, clm_data_dir)

    geom_data = [];
    valid_ids = [];
    
    load('../pca_generation/pdm_68_aligned_wild.mat');
    
    for i=1:numel(users)
        
        geom_files = dir([clm_data_dir, '/' users{i} '*.txt']);
        
        for g=1:numel(geom_files)
            m_file = [clm_data_dir, '/', geom_files(g).name, '.mat'];

            if(~exist(m_file, 'file'))
                res = dlmread([clm_data_dir, '/', geom_files(g).name], ' ');
                res = res(:,[1,2, 3:2:end]);    
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
end