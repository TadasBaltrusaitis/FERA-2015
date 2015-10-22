
features_exe = '"../C++ models/Release/FeatureExtraction.exe"';
find_SEMAINE;


% Go two levels deep
semaine_dirs = train_recs;
out_loc = [SEMAINE_dir, '../processed_data/train/'];

parfor f1=1:numel(semaine_dirs)

    if(isdir([SEMAINE_dir, semaine_dirs{f1}]))
        
        vid_files = dir([SEMAINE_dir, semaine_dirs{f1}, '/*.avi']);

        f1_dir = semaine_dirs{f1};
        
        for v=1:numel(vid_files)

            command = features_exe;

            curr_vid = [SEMAINE_dir, f1_dir, '/', vid_files(v).name];

            name = f1_dir;
            output_file = [out_loc name '/'];

            output_hog = [out_loc name '.hog'];
            output_params = [out_loc name '.params.txt'];

            command = cat(2, command, [' -rigid -f "' curr_vid '" -simalign "' output_file  '" -simscale 0.7 -simsize 112']);
%             command = cat(2, command, [' -rigid -f "' curr_vid '" -simscale 0.7 -simsize 112']);
            command = cat(2, command, [' -hogalign "' output_hog '"']);
            command = cat(2, command, [' -oparams "' output_params '"']);
            dos(command);

        end
    end
end

%%
semaine_dirs = devel_recs;
out_loc = [SEMAINE_dir, '../processed_data/devel/'];

parfor f1=1:numel(semaine_dirs)

    if(isdir([SEMAINE_dir, semaine_dirs{f1}]))
        
        vid_files = dir([SEMAINE_dir, semaine_dirs{f1}, '/*.avi']);

        f1_dir = semaine_dirs{f1};
        
        for v=1:numel(vid_files)

            command = features_exe;

            curr_vid = [SEMAINE_dir, f1_dir, '/', vid_files(v).name];

            name = f1_dir;
            output_file = [out_loc name '/'];

            output_hog = [out_loc name '.hog'];
            output_params = [out_loc name '.params.txt'];

            command = cat(2, command, [' -rigid -f "' curr_vid '" -simalign "' output_file  '" -simscale 0.7 -simsize 112']);
%             command = cat(2, command, [' -rigid -f "' curr_vid '" -simscale 0.7 -simsize 112']);
            command = cat(2, command, [' -hogalign "' output_hog '"']);
            command = cat(2, command, [' -oparams "' output_params '"']);
            dos(command);

        end
    end
end