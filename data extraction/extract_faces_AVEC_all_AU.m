oldDir = chdir('C:\Users\Tadas-local-Quadros\Documents\psyche\Release');

features_exe = '"SimpleCLM.exe"';

avec_loc = 'C:\tadas\face_datasets\avec\';

out_loc = 'C:\tadas\face_datasets\semaine_proc/';

% Go two levels deep
avec_dirs = dir(avec_loc);
avec_dirs = avec_dirs(3:end);

% Just train for now
avec_dirs = avec_dirs([1,3]);

for f1=1:numel(avec_dirs)

    avec_dirs_level_2 = dir([avec_loc, avec_dirs(f1).name]);
    avec_dirs_level_2 = avec_dirs_level_2(3:end);
   
    for f2=1:numel(avec_dirs_level_2)

        vid_files = dir([avec_loc, avec_dirs(f1).name, '/', avec_dirs_level_2(f2).name, '/*.avi']);
        
        f1_dir = avec_dirs(f1).name;
        f2_dir = avec_dirs_level_2(f2).name;
        
        parfor v=1:numel(vid_files)
            
            command = features_exe;
            
            curr_vid = [avec_loc, f1_dir, '/', f2_dir, '/', vid_files(v).name];
            
            [~,name,~] = fileparts(curr_vid);
            output_file = [out_loc name '/'];

            output_aus = [out_loc name '_au.txt'];
    
            command = cat(2, command, [' -f "' curr_vid '"']);
            command = cat(2, command, [' -op "' output_aus '"']);
            dos(command);
            
        end

    end    
    
end