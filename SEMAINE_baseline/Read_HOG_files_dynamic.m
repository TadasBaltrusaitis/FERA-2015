function [hog_data, vid_id] = Read_HOG_files_dynamic(users, hog_data_dir)

    hog_data = [];
    vid_id = {};
    
    feats_filled = 0;

    for i=1:numel(users)
        
        hog_file = [hog_data_dir, 'LeftVideo' users{i} '_comp.hog'];
        
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
        
        % Median normalise the data in a running median way
        
 
        % Do the median computation every 5 frames then faster

        med_file = sprintf('./medians/%s_%s.mat', 'LeftVideo', users{i});
        
        if(~exist(med_file, 'file'))
            meds = zeros(size(curr_data));
                   
            num_every = 5;
            filled_so_far = 0;
            for m=1:num_every:size(curr_data,1)

                if(m==1)
                    curr_med = curr_data(1,:);
                else                
                    curr_med = median(curr_data(1:m,:), 1);
                end

                to_fill = num_every;

                if(to_fill + filled_so_far > size(curr_data,1))
                    to_fill = size(curr_data,1) - filled_so_far;
                end       

                meds(filled_so_far+1:filled_so_far+to_fill,:) = repmat(curr_med, to_fill, 1);
                filled_so_far = filled_so_far + to_fill;            

            end
            
            if(~exist('./medians', 'file'))
                mkdir('./medians/');
            end
            
            save(med_file, 'meds');
        else
           load(med_file); 
        end
        
        curr_data = bsxfun(@plus, curr_data, -meds);
        
        vid_id_curr = cell(curr_ind,1);
        vid_id_curr(:) = users(i);
        
        vid_id = cat(1, vid_id, vid_id_curr);
        
        % Assume same number of frames per video
        if(i==1)
            hog_data = zeros(curr_ind*numel(users), num_feats);
        end
        
        if(size(hog_data,1) < feats_filled+curr_ind)
           hog_data = cat(1, hog_data, zeros(feats_filled + curr_ind - size(hog_data,1), num_feats));
        end
        
        hog_data(feats_filled+1:feats_filled+curr_ind,:) = curr_data;
        
        feats_filled = feats_filled + curr_ind;
        
    end
    
    hog_data = hog_data(1:feats_filled,:);
    
end