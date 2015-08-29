clear
features_exe = '"..\C++ models\Release\FeatureExtraction.exe"';

ck_loc = 'D:\Datasets\ck+\cohn-kanade-images\';

out_loc = 'D:\datasets\face_datasets/hog_aligned_rigid\';
out_loc_params = 'D:\datasets\face_datasets/clm_params\';

% Go two levels deep
ck_dirs = dir(ck_loc);
ck_dirs = ck_dirs(3:end);

parfor f1=1:numel(ck_dirs)

    ck_dirs_level_2 = dir([ck_loc, ck_dirs(f1).name]);
    ck_dirs_level_2 = ck_dirs_level_2(3:end);
   
    for f2=1:numel(ck_dirs_level_2)

        if(~isdir([ck_loc, ck_dirs(f1).name, '/', ck_dirs_level_2(f2).name]))
           continue; 
        end       
        
        command = features_exe;

        curr_vid = [ck_loc, ck_dirs(f1).name, '/', ck_dirs_level_2(f2).name];
        
        name = [ck_dirs(f1).name, '_',  ck_dirs_level_2(f2).name];

        output_file = [out_loc name '/'];

        output_hog = [out_loc name '.hog'];
        output_params = [out_loc_params name '.txt'];
            
        command = cat(2, command, [' -rigid -asvid -fdir "' curr_vid '" -simalign "' output_file  '" -simscale 0.7 -simsize 112 -g']);
        command = cat(2, command, [' -hogalign "' output_hog, '"' ]);
        
        command = cat(2, command, [' -oparams "' output_params '"']);
        
        dos(command);
            
    end    
    
end