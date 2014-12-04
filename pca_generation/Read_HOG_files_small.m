function [hog_data, vid_id] = Read_HOG_files_small(hog_files, hog_data_dir)

    hog_data = [];
    vid_id = {};
    
    feats_filled = 0;

    curr_data_buff = [];
        
    for i=1:numel(hog_files)
        
        hog_file = [hog_data_dir, hog_files(i).name];
        
        f = fopen(hog_file, 'r');
                         
        curr_ind = 0;
        
        while(~feof(f))
                        
            if(i == 1 && curr_ind == 0)
                num_cols = fread(f, 1, 'int32');
                if(isempty(num_cols))
                    break;
                end

                num_rows = fread(f, 1, 'int32');
                num_chan = fread(f, 1, 'int32');

                curr_ind = curr_ind + 1;            

                % preallocate some space
                if(curr_ind == 1)
                    curr_data_buff = zeros(5000, num_rows * num_cols * num_chan);
                    num_feats =  num_rows * num_cols * num_chan;
                end

                if(curr_ind > size(curr_data_buff,1))
                    curr_data_buff = cat(1, curr_data_buff, zeros(6000, num_rows * num_cols * num_chan));
                end
                feature_vec = fread(f, [1, num_rows * num_cols * num_chan], 'float32');
                curr_data_buff(curr_ind, :) = feature_vec;
            else
            
                % Reading in batches of 5000
                
                feature_vec = fread(f, [3 + num_rows * num_cols * num_chan, 5000], 'float32');
                feature_vec = feature_vec(4:end,:)';
                
                num_rows_read = size(feature_vec,1);
                
                curr_data_buff(curr_ind+1:curr_ind+num_rows_read,:) = feature_vec;
                
                curr_ind = curr_ind + size(feature_vec,1);
                
            end
                        
        end
        
        fclose(f);
        
        curr_data_small = curr_data_buff(1:curr_ind,:);
        vid_id_curr = cell(curr_ind,1);
        vid_id_curr(:) = {hog_files(i).name};
        
        % Keep up to 20 frames from the whole video (so that it is balanced
        % per dataset/video/participant)
        
        num_instances = 20;
        
        increment = round(curr_ind / num_instances);
        if(increment == 0)
            increment = 1;
        end
        
        curr_data_small = curr_data_small(1:increment:end,:);
        vid_id_curr = vid_id_curr(1:increment:end,:);        
        
        vid_id = cat(1, vid_id, vid_id_curr);
        
        % Assume same number of frames per video
        if(i==1)
            hog_data = zeros(10*numel(hog_files), num_feats);
        end
        
        if(size(hog_data,1) < feats_filled+size(curr_data_small,1))
           hog_data = cat(1, hog_data, zeros(feats_filled + size(curr_data_small,1) - size(hog_data,1), num_feats));
        end
        
        hog_data(feats_filled+1:feats_filled + size(curr_data_small,1),:) = curr_data_small;
        
        feats_filled = feats_filled + size(curr_data_small,1);
        
    end
    
    hog_data = hog_data(1:feats_filled,:);
    
end