clear
features_exe = '"..\C++ models\Release\FeatureExtraction.exe"';

unbc_loc = 'D:\Datasets\UNBC\Images/';

out_loc = 'D:\Datasets\face_datasets/';

% Go two levels deep
unbc_dirs = dir(unbc_loc);
unbc_dirs = unbc_dirs(3:end);

if(~exist([out_loc, '/clm_params/'], 'file'))
    mkdir([out_loc, '/clm_params/']);   
end

parfor f1=1:numel(unbc_dirs)

    unbc_dirs_level_2 = dir([unbc_loc, unbc_dirs(f1).name]);
    unbc_dirs_level_2 = unbc_dirs_level_2(3:end);
   
    for f2=1:numel(unbc_dirs_level_2)

        if(~isdir([unbc_loc, unbc_dirs(f1).name, '/', unbc_dirs_level_2(f2).name]))
           continue; 
        end       
        
        command = features_exe;

        curr_vid = [unbc_loc, unbc_dirs(f1).name, '/', unbc_dirs_level_2(f2).name];
        
        name = [unbc_dirs(f1).name, '_',  unbc_dirs_level_2(f2).name];

        output_file = [out_loc, '/hog_aligned_rigid/', name '/'];

        output_hog = [out_loc, '/hog_aligned_rigid/', name '.hog'];
        output_params = [out_loc, '/clm_params/', name '.txt'];
        
        command = cat(2, command, [' -rigid -asvid -fdir "' curr_vid '" -simalign "' output_file  '" -simscale 0.7 -simsize 112 -g']);
        command = cat(2, command, [' -hogalign "' output_hog '"']);
        
        command = cat(2, command, [' -oparams "' output_params '"']);
    
        dos(command);
            
    end    
    
end