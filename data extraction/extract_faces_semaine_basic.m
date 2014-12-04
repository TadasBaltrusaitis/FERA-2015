oldDir = chdir('../C++ models/Release');

features_exe = '"FeatureExtraction.exe"';

semaine_loc = 'I:/datasets/FERA_2015/Semaine/SEMAINE-Sessions/';

out_loc = 'I:/datasets/FERA_2015/Semaine/processed_data_big/';

% Go two levels deep
semaine_dirs = dir(semaine_loc);
semaine_dirs = semaine_dirs(3:end);

for f1=1:numel(semaine_dirs)

    if(isdir([semaine_loc, semaine_dirs(f1).name]))
        
        vid_files = dir([semaine_loc, semaine_dirs(f1).name, '/*.avi']);

        f1_dir = semaine_dirs(f1).name;
        
        for v=1:numel(vid_files)

            command = features_exe;

            curr_vid = [semaine_loc, f1_dir, '/', vid_files(v).name];

            name = f1_dir;
            output_file = [out_loc name '/'];

            output_hog = [out_loc name '.hog'];
            output_params = [out_loc name '.params.txt'];
            output_aus = [out_loc name '.au.txt'];
            output_neut = [out_loc name '.neutral'];

            command = cat(2, command, [' -rigid -f "' curr_vid '" -simalign "' output_file  '" -simscale 0.7 -simsize 112']);
            command = cat(2, command, [' -hogalign "' output_hog '"']);
            command = cat(2, command, [' -oparams "' output_params '"']);
            dos(command);

        end
    end
end

chdir(oldDir)