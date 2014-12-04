oldDir = chdir('C:\Users\Tadas-local-Quadros\Documents\CLM-framework\Release');

features_exe = '"FeatureExtraction.exe"';

ck_loc = 'C:\tadas\face_datasets\ck+\cohn-kanade-images\';

out_loc = 'C:\tadas\face_datasets\hog_aligned\';

% Go two levels deep
ck_dirs = dir(ck_loc);
ck_dirs = ck_dirs(3:end);

for f1=1:numel(ck_dirs)

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

        if(exist(output_hog, 'file'))
            continue;    
        end

        command = cat(2, command, [' -asvid -fdir "' curr_vid '" -simalign "' output_file  '" -simscale 0.6 -simsize 96 -g']);
        command = cat(2, command, [' -hogalign "' output_hog ]);
        dos(command);
            
    end    
    
end