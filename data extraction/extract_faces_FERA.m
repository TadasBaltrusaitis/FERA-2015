oldDir = chdir('C:\Users\Tadas-local-Quadros\Documents\CLM-framework\Release');

features_exe = '"FeatureExtraction.exe"';

fera_loc = 'C:\tadas\face_datasets\fera\';

out_loc = 'C:\tadas\face_datasets\hog_aligned\';

% Go two levels deep
fera_dirs = dir(fera_loc);
fera_dirs = fera_dirs(3:end);

for f1=1:numel(fera_dirs)

    fera_dirs_level_2 = dir([fera_loc, fera_dirs(f1).name]);
    fera_dirs_level_2 = fera_dirs_level_2(3:end);
   
    for f2=1:numel(fera_dirs_level_2)

        vid_files = dir([fera_loc, fera_dirs(f1).name, '/', fera_dirs_level_2(f2).name, '/*.avi']);
        
        for v=1:numel(vid_files)
            
            command = features_exe;
            
            curr_vid = [fera_loc, fera_dirs(f1).name, '/', fera_dirs_level_2(f2).name, '/', vid_files(v).name];
            
            [~,name,~] = fileparts(curr_vid);
            output_file = [out_loc name '/'];

            output_hog = [out_loc name '.hog'];
    
            if(exist(output_hog, 'file'))
                output_file = [out_loc name '_1/'];
                output_hog = [out_loc name '_1.hog'];      
            end
            
            command = cat(2, command, [' -f "' curr_vid '" -simalign "' output_file  '" -simscale 0.6 -simsize 96 -g']);
            command = cat(2, command, [' -hogalign "' output_hog ]);
            dos(command);
            
        end

    end    
    
end