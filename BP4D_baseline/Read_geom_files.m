function [geom_data, valid_ids] = Read_geom_files(users, hog_data_dir)

    geom_data = [];
    valid_ids = [];

    load('../pca_generation/pdm_68_aligned_wild.mat');

    brow_ids = [18,27,19,26,20,25,21,24,22,23];
    
    % TODO potentially reorder these?
%     left_eye_ids = [18:22, 37:42];
%     right_eye_ids = [23:27, 43:48];
    
    left_eye_ids = [37,18,38,19,39,20,40,21,41,22,42];
    right_eye_ids = [43,23,44,24,45,25,46,26,49,27,48];
    
    eye_ids = [37,38,42,39,41,40,43,44,48,45,47,46];
%     nose_ids = 28:36;
    
    nose_ids = [28,32,29,33,30,34,31,35,36];
    
    out_ids = 1:17;
%     lip_ids = 49:68;
    
    out_lip_ids = [49,50,60,51,59,52,58,53,57,54,56,55];
    in_lip_ids = [61,62,68,63,67,64,66,65];
    
%     lip_ids = 49:68;
    
    for i=1:numel(users)
        
        geom_files = dir([hog_data_dir, '/train/', users{i} '*.params.txt']);
        geom_dir = [hog_data_dir, '/train/'];
        if(isempty(geom_files))
            geom_files = dir([hog_data_dir, '/devel/', users{i} '*.params.txt']);
            geom_dir = [hog_data_dir, '/devel/'];
        end
        
        for h=1:numel(geom_files)
            geom_file = [geom_dir, geom_files(h).name];
            [~, nm, ~] = fileparts(geom_file);
            m_file = [geom_dir, '/' nm '.params.mat'];
              
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
            
            % Compute differences between locations
%             XS = actual_locs(:,1:end/3);
%             YS = actual_locs(:,end/3+1:2*end/3);
%             ZS = actual_locs(:,2*end/3:end);
% 
%             brows = cat(3, XS(:,brow_ids), YS(:,brow_ids), ZS(:,brow_ids));
%             eyes = cat(3, XS(:,eye_ids), YS(:,eye_ids), ZS(:,eye_ids));
%             nose = cat(3, XS(:,nose_ids), YS(:,nose_ids), ZS(:,nose_ids));
%             left_eye = cat(3, XS(:,left_eye_ids), YS(:,left_eye_ids), ZS(:,left_eye_ids));        
%             right_eye = cat(3, XS(:,right_eye_ids), YS(:,right_eye_ids), ZS(:,right_eye_ids));            
%             in_lips = cat(3, XS(:,in_lip_ids), YS(:,in_lip_ids), ZS(:,in_lip_ids));
%             out_lips = cat(3, XS(:,out_lip_ids), YS(:,out_lip_ids), ZS(:,out_lip_ids));
%             outline = cat(3, XS(:,out_ids), YS(:,out_ids), ZS(:,out_ids));
%             
%             l_eye_dists = euclidean_distances(left_eye);
%             r_eye_dists = euclidean_distances(right_eye);
%             brows_dists = euclidean_distances(brows);
%             eyes_dists = euclidean_distances(eyes);
%             nose_dists = euclidean_distances(nose);
%             in_lip_dists = euclidean_distances(in_lips);
%             out_lip_dists = euclidean_distances(out_lips);
%             out_dists = euclidean_distances(outline);
%             
%             angles_brows = vector_angles_2D(brows);
%             angles_l_eye = vector_angles_2D(left_eye);
%             angles_r_eye = vector_angles_2D(right_eye);
%             angles_eyes = vector_angles_2D(eyes);
%             angles_nose = vector_angles_2D(nose);
%             angles_in_lips = vector_angles_2D(in_lips);
%             angles_out_lips = vector_angles_2D(out_lips);
%             angles_out = vector_angles_2D(outline);
%             res = cat(2, res, brows_dists, l_eye_dists, r_eye_dists, eyes_dists, nose_dists, in_lip_dists, out_lip_dists, angles_brows, angles_eyes, angles_l_eye, angles_r_eye, angles_nose, angles_in_lips, angles_out_lips);
          
%             res = cat(2, brows_dists, l_eye_dists, r_eye_dists, eyes_dists, nose_dists, in_lip_dists, out_lip_dists, angles_brows, angles_eyes, angles_l_eye, angles_r_eye, angles_nose, angles_in_lips, angles_out_lips);
          
%             res = cat(2, brows_dists, l_eye_dists, r_eye_dists, eyes_dists, nose_dists, in_lip_dists, out_lip_dists, out_dists);
          
%             res = cat(2, angles_brows, angles_eyes, angles_l_eye, angles_r_eye, angles_nose, angles_in_lips, angles_out_lips, angles_out);
          
%             res = cat(2, res, brows_dists, l_eye_dists, r_eye_dists, eyes_dists, nose_dists, in_lip_dists, out_lip_dists, out_dists, angles_brows, angles_eyes, angles_l_eye, angles_r_eye, angles_nose, angles_in_lips, angles_out_lips, angles_out);
          
            valid_ids = cat(1, valid_ids, valid);
            
            geom_data = cat(1, geom_data, res);
                
        end
    end
end

function [dists] = euclidean_distances(data)

    dists = (data(:,1:end,:) - data(:,[2:end,1], :)).^2;
    dists_2 = (data(:,1:end,:) - data(:,[3:end,1,2], :)).^2;
    dists = sqrt(dists(:,:,1) + dists(:,:,2) + dists(:,:,3));
    dists_2 = sqrt(dists_2(:,:,1) + dists_2(:,:,2) + dists_2(:,:,3));
    dists = cat(2, dists, dists_2);
    
end

function angles = vector_angles_3D(data)

    % Construct the vectors
    vecs_1 = data(:,[2:end,1],:) - data(:,1:end,:);
    vecs_2 = data(:,[3:end,1,2],:) - data(:,1:end,:);
    
    size_orig = size(data);
    
    % Reshape the vectors
    vecs_1 = reshape(vecs_1, size(vecs_1,1)*size(vecs_1,2), 3);
    vecs_2 = reshape(vecs_2, size(vecs_2,1)*size(vecs_2,2), 3);
    % compute the cross products
    crosses = cross(vecs_1, vecs_2);
    crosses = sqrt(sum(crosses.^2,2));
    
    % Compute the dot products
    dots = sum(vecs_1 .* vecs_2,2);
       
    angles = atan2(crosses, dots);

    angles = reshape(angles, size_orig(1), size_orig(2));

    vecs_1 = data(:,[3:end,1,2],:) - data(:,1:end,:);
    vecs_2 = data(:,[4:end,1,2,3],:) - data(:,1:end,:);
    
    size_orig = size(data);
    
    % Reshape the vectors
    vecs_1 = reshape(vecs_1, size(vecs_1,1)*size(vecs_1,2), 3);
    vecs_2 = reshape(vecs_2, size(vecs_2,1)*size(vecs_2,2), 3);
    % compute the cross products
    crosses = cross(vecs_1, vecs_2);
    crosses = sqrt(sum(crosses.^2,2));
    
    % Compute the dot products
    dots = sum(vecs_1 .* vecs_2,2);
       
    angles_3 = atan2(crosses, dots);

    angles_3 = reshape(angles3, size_orig(1), size_orig(2));    
    
    angles = cat(2, angles, angles_3);
end

function angles = vector_angles_2D(data)

    data = data(:,:,1:2);
    
    % Construct the vectors
    vecs_1 = data(:,[2:end,1],:) - data(:,1:end,:);
    vecs_2 = data(:,[3:end,1,2],:) - data(:,1:end,:);
    
    size_orig = size(data);
    
    % Reshape the vectors
    vecs_1 = reshape(vecs_1, size(vecs_1,1)*size(vecs_1,2), 2);    
    vecs_2 = reshape(vecs_2, size(vecs_2,1)*size(vecs_2,2), 2);
    
    norms_1 = sqrt(sum(vecs_1.^2,2));
    norms_2 = sqrt(sum(vecs_2.^2,2));
    
    vecs_1 = bsxfun(@times, vecs_1, 1./norms_1);
    vecs_2 = bsxfun(@times, vecs_2, 1./norms_2);    
    
    angles1 = atan2(vecs_1(:,2), vecs_1(:,1));
    angles1(angles1 > pi/2) = angles1(angles1 > pi/2) - pi;
    angles1(angles1 < -pi/2) = angles1(angles1 < -pi/2) + pi;
    
    angles2 = atan2(vecs_2(:,2), vecs_2(:,1));
    angles2(angles2 > pi/2) = angles2(angles2 > pi/2) - pi;
    angles2(angles2 < -pi/2) = angles2(angles2 < -pi/2) + pi;
    
    angles = angles1 - angles2;
    angles = reshape(angles, size_orig(1), size_orig(2));
        
    % Construct the vectors
    vecs_1 = data(:,[3:end,1,2],:) - data(:,1:end,:);
    vecs_2 = data(:,[4:end,1,2,3],:) - data(:,1:end,:);
    
    size_orig = size(data);
    
    % Reshape the vectors
    vecs_1 = reshape(vecs_1, size(vecs_1,1)*size(vecs_1,2), 2);    
    vecs_2 = reshape(vecs_2, size(vecs_2,1)*size(vecs_2,2), 2);
    
    norms_1 = sqrt(sum(vecs_1.^2,2));
    norms_2 = sqrt(sum(vecs_2.^2,2));
    
    vecs_1 = bsxfun(@times, vecs_1, 1./norms_1);
    vecs_2 = bsxfun(@times, vecs_2, 1./norms_2);    
    
    angles1 = atan2(vecs_1(:,2), vecs_1(:,1));
    angles1(angles1 > pi/2) = angles1(angles1 > pi/2) - pi;
    angles1(angles1 < -pi/2) = angles1(angles1 < -pi/2) + pi;
    
    angles2 = atan2(vecs_2(:,2), vecs_2(:,1));
    angles2(angles2 > pi/2) = angles2(angles2 > pi/2) - pi;
    angles2(angles2 < -pi/2) = angles2(angles2 < -pi/2) + pi;
    
    angles_3 = angles1 - angles2;
    angles_3 = reshape(angles_3, size_orig(1), size_orig(2));

    angles = cat(2, angles, angles_3);
end