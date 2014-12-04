oldDir = chdir('C:\Users\Tadas-local-Quadros\Documents\CLM-framework\Release');

features_exe = '"FeatureExtraction.exe"';

avec_loc = 'C:\tadas\face_datasets\avec\';

out_loc = 'C:\tadas\face_datasets\hog_aligned_rigid\';

% Go two levels deep
avec_dirs = dir(avec_loc);
avec_dirs = avec_dirs(3:end);

parfor f1=1:numel(avec_dirs)

    avecf_dirs_level_2 = dir([avec_loc, avec_dirs(f1).name]);
    avecf_dirs_level_2 = avecf_dirs_level_2(3:end);
   
    for f2=1:numel(avecf_dirs_level_2)

        vid_files = dir([avec_loc, avec_dirs(f1).name, '/', avecf_dirs_level_2(f2).name, '/*.avi']);
        
        % As there are roughly four vids per person save some space
        for v=1:4:numel(vid_files)
            
            command = features_exe;
            
            curr_vid = [avec_loc, avec_dirs(f1).name, '/', avecf_dirs_level_2(f2).name, '/', vid_files(v).name];
            
            [~,name,~] = fileparts(curr_vid);
            output_file = [out_loc name '/'];

            output_hog = [out_loc name '.hog'];
    
            if(exist(output_hog, 'file'))
                output_file = [out_loc name '_1/'];
                output_hog = [out_loc name '_1.hog'];      
            end
            
            command = cat(2, command, [' -rigid -f "' curr_vid '" -simalign "' output_file  '" -simscale 0.6 -simsize 96 -g']);
            command = cat(2, command, [' -hogalign "' output_hog ]);
            dos(command);
            
        end

    end    
    
end