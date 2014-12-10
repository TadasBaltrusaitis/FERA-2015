function [ labels, valid_ids, vid_ids, filenames  ] = extract_BP4D_labels_intensity( BP4D_dir, recs, aus )
%EXTRACT_SEMAINE_LABELS Summary of this function goes here
%   Detailed explanation goes here
    
    aus_BP4D = [6, 10, 14, 17];
    
    inds_to_use = [];
    
    for i=1:numel(aus)
    
        inds_to_use = cat(1, inds_to_use, find(aus_BP4D == aus(i)));
        
    end
    
    % Need to find relevant files for the relevant user and for the
    % relevant AU
    files_all = dir([BP4D_dir, '/*.csv']);
    num_files = numel(files_all);
    
    labels = cell(num_files, 1);
    valid_ids = cell(num_files, 1);
    vid_ids = zeros(num_files, 2);
    filenames = cell(num_files, 1);
    
    file_id = 1;
    
    for i=1:numel(recs)

        for au=aus
        
            % Dealing with naming inconsistency
            rec_short = recs{i}(1,3,4,5);
            
            % Find the relevant files
            sprintf('%s/../BP4D-AUIntensityCodes/Project/AU%02d/*%s*'
            rel_files = dir([BP4D_dir, '/Project/'\BP4D-AUIntensityCodes

            csvs = dir([BP4D_dir, '/', recs{i}, '*.csv']);

            for f=1:numel(csvs)

                file = [BP4D_dir, '/', csvs(f).name];

                [~, filename,~] = fileparts(file);
                filenames{file_id} = filename;

                OCC = csvread(file); %import annotations for one video file
                frame_nums = OCC(2:end,1); %get all frame numbers
                codes = OCC(2:end,2:end); %get codes for all action units
                occlusions = OCC(2:end,end);

                codes = codes(:, aus_BP4D);

                % Finding the invalid regions
                valid = occlusions ~= 1;

                for s=1:size(codes,2)

                    valid = valid & codes(:,s) ~= 9;

                end

                vid_ids(file_id,:) = [frame_nums(1), frame_nums(end)];

                labels{file_id} = codes(:, inds_to_use);

                % all indices in SEMAINE are valid
                valid_ids{file_id} = valid;

                file_id = file_id + 1;
            end 
        
        end
    end
    
    labels = labels(1:file_id-1);
    valid_ids = valid_ids(1:file_id-1);
    vid_ids = vid_ids(1:file_id-1, :);
    filenames = filenames(1:file_id-1);
    
end

