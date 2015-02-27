function [geom_data, valid_ids] = Read_geom_files_dynamic(users, hog_data_dir)

    geom_data = [];
    valid_ids = [];
    
    load('../pca_generation/pdm_68_multi_pie.mat');
    
    for i=1:numel(users)
        
        geom_files = dir([hog_data_dir, users{i} '*.params.txt']);
        geom_data_curr = [];
        for h=1:numel(geom_files)
            geom_file = [hog_data_dir, geom_files(h).name];
                        
            [~, nm, ~] = fileparts(geom_file);
            m_file = [hog_data_dir, '/' nm '.params.mat'];
              
            if(~exist(m_file, 'file'))
                res = dlmread(geom_file, ' ');
                res = res(:,[1,2, 3:2:end]);       
                save(m_file, 'res');
            else
                load(m_file);
            end
            
            valid = res(:,2);      
            res = res(:, 9:end);
                    
            % TODO remove?
%             actual_locs = bsxfun(@plus, res * V', M');
            actual_locs = res * V';
            res = cat(2, actual_locs, res);
            
            valid_ids = cat(1, valid_ids, valid);            
            
%             actual_locs = bsxfun(@plus, res * V', M');
%             res = actual_locs;            
%             
            geom_data_curr = cat(1, geom_data_curr, res);
        end
        geom_data_curr = bsxfun(@plus, geom_data_curr, -median(res));
        
        geom_data = cat(1, geom_data, geom_data_curr);
                
    end
end