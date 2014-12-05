function [hog_data] = Read_HOG_files_CK(users, hog_data_dir)

    hog_data = zeros(numel(users)*2, 3100);
    
    for i=1:numel(users)
        
        hog_file = [hog_data_dir, '/' users{i} '.hog'];
        
        f = fopen(hog_file, 'r');
                          
        curr_data = [];
        curr_ind = 0;
        
        while(~feof(f))
                        
            if(curr_ind == 0)
                num_cols = fread(f, 1, 'int32');
                if(isempty(num_cols))
                    break;
                end

                num_rows = fread(f, 1, 'int32');
                num_chan = fread(f, 1, 'int32');

                curr_ind = curr_ind + 1;            

                % preallocate some space
                if(curr_ind == 1)
                    curr_data = zeros(5000, num_rows * num_cols * num_chan);
                    num_feats =  num_rows * num_cols * num_chan;
                end

                if(curr_ind > size(curr_data,1))
                    curr_data = cat(1, curr_data, zeros(6000, num_rows * num_cols * num_chan));
                end
                feature_vec = fread(f, [1, num_rows * num_cols * num_chan], 'float32');
                curr_data(curr_ind, :) = feature_vec;
            else
            
                % Reading in batches of 5000
                
                feature_vec = fread(f, [3 + num_rows * num_cols * num_chan, 5000], 'float32');
                feature_vec = feature_vec(4:end,:)';
                
                num_rows_read = size(feature_vec,1);
                
                curr_data(curr_ind+1:curr_ind+num_rows_read,:) = feature_vec;
                
                curr_ind = curr_ind + size(feature_vec,1);
                
            end
                        
        end
        
        fclose(f);
        
        curr_data = curr_data(1:curr_ind,:);
                
        % Take the first and the last frames
        curr_data = cat(1, curr_data(1,:), curr_data(end,:));
                
        hog_data((i-1)*2+1:(i-1)*2+2, :) = curr_data;
        
        
    end
    
    
end