function [geom_data] = Read_geom_files_dynamic(users, hog_data_dir)

    geom_data = [];
    
    load('../../pca_generation/pdm_68_aligned_wild.mat');
    
    for i=1:numel(users)
        
        geom_file = [hog_data_dir, '/../clm_params/LeftVideo' users{i} '_comp.txt'];        
        
        res = dlmread(geom_file, ' ');
        res = res(:,15:2:end);       
        
        actual_locs = res * V';
        res = cat(2, actual_locs, res);
        
        res = bsxfun(@plus, res, -median(res));        
        
        geom_data = cat(1, geom_data, res);
                
    end
end