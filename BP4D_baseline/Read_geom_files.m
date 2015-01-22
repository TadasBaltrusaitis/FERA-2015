function [geom_data] = Read_geom_files(users, hog_data_dir)

    geom_data = [];

    for i=1:numel(users)
        
        geom_files = dir([hog_data_dir, users{i} '*.params.txt']);
        
        for h=1:numel(geom_files)
            geom_file = [hog_data_dir, geom_files(h).name];

            res = dlmread(geom_file, ' ');
            res = res(:,15:2:end);       

            geom_data = cat(1, geom_data, res);
                
        end
    end
end