function [params_data, vid_id] = Read_CLM_files_run_med(users, clm_data_dir)

    params_data = [];
    vid_id = {};
    
    for i=1:numel(users)
        
        params_file = [clm_data_dir, 'LeftVideo' users{i} '_comp.txt'];
        
        params_data_curr = dlmread(params_file);
        params_data_curr = params_data_curr(:, 9:end);
        
        med_file = sprintf('../clm_medians/%s_%s.mat', 'LeftVideo', users{i});
        
        if(~exist(med_file, 'file'))
            meds = zeros(size(params_data_curr));

            num_every = 5;
            filled_so_far = 0;
            for m=1:num_every:size(params_data_curr,1)

                if(m==1)
                    curr_med = params_data_curr(1,:);
                else                
                    curr_med = median(params_data_curr(1:m,:), 1);
                end

                to_fill = num_every;

                if(to_fill + filled_so_far > size(params_data_curr,1))
                    to_fill = size(params_data_curr,1) - filled_so_far;
                end       

                meds(filled_so_far+1:filled_so_far+to_fill,:) = repmat(curr_med, to_fill, 1);
                filled_so_far = filled_so_far + to_fill;            

            end
            save(med_file, 'meds');
        else
           load(med_file); 
        end
        params_data_curr = bsxfun(@plus, params_data_curr, -meds);
        
        vid_id_curr = cell(size(params_data_curr, 1),1);
        vid_id_curr(:) = users(i);
        
        vid_id = cat(1, vid_id, vid_id_curr);
                
        params_data = cat(1, params_data, params_data_curr);
    end
        
end