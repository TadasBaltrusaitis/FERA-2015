oldDir = chdir('C:\Users\Tadas-local-Quadros\Documents\CLM-framework\Release');

features_exe = '"FeatureExtraction.exe"';

fera_loc = 'C:\tadas\face_datasets\fera\';

out_loc = 'C:\tadas\face_datasets\hog_aligned_rigid\';

% Go two levels deep
fera_dirs = dir(fera_loc);
fera_dirs = fera_dirs(3:end);

parfor f1=1:numel(fera_dirs)

    fera_dirs_level_2 = dir([fera_loc, fera_dirs(f1).name]);
    fera_dirs_level_2 = fera_dirs_level_2(3:end);
   
    for f2=1:numel(fera_dirs_level_2)

        vid_files = dir([fera_loc, fera_dirs(f1).name, '/', fera_dirs_level_2(f2).name, '/*.avi']);
        
        for v=1:numel(vid_files)
            
            command = features_exe;
            
            curr_vid = [fera_loc, fera_dirs(f1).name, '/', fera_dirs_level_2(f2).name, '/', vid_files(v).name];
            
            [~,name,~] = fileparts(curr_vid);
            output_file = [out_loc fera_dirs(f1).name '_' name '/'];

            output_hog = [out_loc fera_dirs(f1).name '_' name '.hog'];
    
            command = cat(2, command, [' -rigid -f "' curr_vid '" -simalign "' output_file  '" -simscale 0.6 -simsize 96 -g']);
            command = cat(2, command, [' -hogalign "' output_hog ]);
            dos(command);
            
        end

    end    
    
end

chdir(oldDir)