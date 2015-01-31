function [geom_data] = Read_geom_files_dynamic(users, vid_ids, hog_data_dir)

    geom_data = [];

    for i=1:numel(users)
        
        geom_file = [hog_data_dir, '/' users{i} '.params.txt'];
        m_file = [hog_data_dir, '/' users{i} '.params.mat'];
        
        if(~exist(m_file, 'file'))
            res = dlmread(geom_file, ' ');
            res = res(vid_ids(i,1)+1:vid_ids(i,2),1:2:end);       
            save(m_file, 'res');
        else
            load(m_file);
        end
        
        res = res(:, 8:end);
        
        res = bsxfun(@plus, res, -median(res));
        
        geom_data = cat(1, geom_data, res);
                
    end
end