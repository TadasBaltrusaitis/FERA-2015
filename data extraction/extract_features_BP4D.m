clear

features_exe = '"../C++ models/Release/FeatureExtraction.exe"';

find_BP4D;
BP4D_dir = [BP4D_dir '\..\BP4D-training\'];

bp4d_dirs = train_recs;
out_loc = [BP4D_dir '\..\processed_data\train\'];
parfor f1=1:numel(bp4d_dirs)

    if(isdir([BP4D_dir, bp4d_dirs{f1}]))
        
        bp4d_2_dirs = dir([BP4D_dir, bp4d_dirs{f1}]);
        bp4d_2_dirs = bp4d_2_dirs(3:end);
        
        f1_dir = bp4d_dirs{f1};
        
        for f2=1:numel(bp4d_2_dirs)
            f2_dir = bp4d_2_dirs(f2).name;
            if(isdir([BP4D_dir, bp4d_dirs{f1}]))
                command = features_exe;

                curr_vid = [BP4D_dir, f1_dir, '/', f2_dir, '/'];

                name = [f1_dir '_' f2_dir];
                output_file = [out_loc name '/'];

                output_hog = [out_loc name '.hog'];
                output_params = [out_loc name '.params.txt'];
%                 output_aus = [out_loc name '.au.txt'];
%                 output_neut = [out_loc name '.neutral'];
%                 command = cat(2, command, [' -fx 2000 -fy 2000 -rigid -asvid -fdir "' curr_vid '" -simalign "' output_file  '" -simscale 0.7 -simsize 112']);
                command = cat(2, command, [' -fx 2000 -fy 2000 -rigid -asvid -fdir "' curr_vid '" -simscale 0.7 -simsize 112']);                
                command = cat(2, command, [' -hogalign "' output_hog '"']);
                command = cat(2, command, [' -oparams "' output_params '"']);
                dos(command);
            end
        end
    end
end

bp4d_dirs = devel_recs;
out_loc = [BP4D_dir '\..\processed_data\devel\'];
parfor f1=1:numel(bp4d_dirs)

    if(isdir([BP4D_dir, bp4d_dirs{f1}]))
        
        bp4d_2_dirs = dir([BP4D_dir, bp4d_dirs{f1}]);
        bp4d_2_dirs = bp4d_2_dirs(3:end);
        
        f1_dir = bp4d_dirs{f1};
        
        for f2=1:numel(bp4d_2_dirs)
            f2_dir = bp4d_2_dirs(f2).name;
            if(isdir([BP4D_dir, bp4d_dirs{f1}]))
                command = features_exe;

                curr_vid = [BP4D_dir, f1_dir, '/', f2_dir, '/'];

                name = [f1_dir '_' f2_dir];
                output_file = [out_loc name '/'];

                output_hog = [out_loc name '.hog'];
                output_params = [out_loc name '.params.txt'];
%                 output_aus = [out_loc name '.au.txt'];
%                 output_neut = [out_loc name '.neutral'];
%                 command = cat(2, command, [' -fx 2000 -fy 2000 -rigid -asvid -fdir "' curr_vid '" -simalign "' output_file  '" -simscale 0.7 -simsize 112']);
                command = cat(2, command, [' -fx 2000 -fy 2000 -rigid -asvid -fdir "' curr_vid '" -simscale 0.7 -simsize 112']);                
                command = cat(2, command, [' -hogalign "' output_hog '"']);
                command = cat(2, command, [' -oparams "' output_params '"']);
                dos(command);
            end
        end
    end
end