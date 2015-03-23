function [geom_data, valid_ids] = Read_geom_files_dynamic(users, vid_ids, hog_data_dir)

    geom_data = [];
    valid_ids = [];
    
    load('../pca_generation/pdm_68_aligned_wild.mat');
    
    brow_ids = 18:27;
    
    % TODO potentially reorder these?
%     left_eye_ids = [18:22, 37:42];
%     right_eye_ids = [23:27, 43:48];
    
    left_eye_ids = [37,18,38,19,39,20,40,21,41,22,42];
    right_eye_ids = [43,23,44,24,45,25,46,26,49,27,48];
    
    eye_ids = 37:48;    
    nose_ids = 28:36;
    
%     out_ids = 1:17;
    
    lip_ids = 49:68;
    
    
    for i=1:numel(users)
        
        geom_file = [hog_data_dir, '/' users{i} '.params.txt'];
        m_file = [hog_data_dir, '/' users{i} '.params.mat'];
        
        if(~exist(m_file, 'file'))
            res = dlmread(geom_file, ' ');
            res = res(vid_ids(i,1)+1:vid_ids(i,2),[1,2, 3:2:end]);       
            save(m_file, 'res');
        else
            load(m_file);
        end
        
        valid = res(:, 2);
        res = res(:, 9:end);
        
        
%         actual_locs = bsxfun(@plus, res * V', M');
        actual_locs = res * V';
        res = cat(2, actual_locs, res);
%         res = actual_locs;
%                
%         % Compute differences between locations
%         XS = actual_locs(:,1:end/3);
%         YS = actual_locs(:,end/3+1:2*end/3);
%         ZS = actual_locs(:,2*end/3:end);
%         
%         brows = cat(3, XS(:,brow_ids), YS(:,brow_ids), ZS(:,brow_ids));
%         eyes = cat(3, XS(:,eye_ids), YS(:,eye_ids), ZS(:,eye_ids));
%         nose = cat(3, XS(:,nose_ids), YS(:,nose_ids), ZS(:,nose_ids));
%         left_eye = cat(3, XS(:,left_eye_ids), YS(:,left_eye_ids), ZS(:,left_eye_ids));        
%         right_eye = cat(3, XS(:,right_eye_ids), YS(:,right_eye_ids), ZS(:,right_eye_ids));
%         lips = cat(3, XS(:,lip_ids), YS(:,lip_ids), ZS(:,lip_ids));
%         
%         l_eye_dists = euclidean_distances(left_eye);
%         r_eye_dists = euclidean_distances(right_eye);
%         brows_dists = euclidean_distances(brows);
%         eyes_dists = euclidean_distances(eyes);
%         nose_dists = euclidean_distances(nose);
%         lip_dists = euclidean_distances(lips);
%         
%         %res = cat(2, res, l_eye_dists, r_eye_dists, brows_dists, eyes_dists, nose_dists, lip_dists);
%         res = cat(2, l_eye_dists, r_eye_dists);
%         
        valid_ids = cat(1, valid_ids, valid);
                
        res = bsxfun(@plus, res, -median(res));

        geom_data = cat(1, geom_data, res);
                
    end
end

function [dists] = euclidean_distances(data)

    dists = (data(:,1:end,:) - data(:,[2:end,1], :)).^2;
    dists = sqrt(dists(:,:,1) + dists(:,:,2) + dists(:,:,3));
    
end