clear

addpath('../data extraction/');
find_BP4D;

bp4d_loc = [BP4D_dir '../BP4D/BP4D-training/'];

out_loc = '../../runners/out_bp4d_static/';

if(~exist(out_loc, 'dir'))
    mkdir(out_loc);
end

oldDir = chdir('../C++ models/Release');

features_exe = '"AUPrediction.exe"';

% Go two levels deep
bp4d_dirs = dir(bp4d_loc);
bp4d_dirs = bp4d_dirs(3:end);
        
parfor f1=1:numel(bp4d_dirs)

    if(isdir([bp4d_loc, bp4d_dirs(f1).name]))
        
        bp4d_2_dirs = dir([bp4d_loc, bp4d_dirs(f1).name]);
        bp4d_2_dirs = bp4d_2_dirs(3:end);
        
        f1_dir = bp4d_dirs(f1).name;

        command = [features_exe, ' -scaling 0.4 -auloc "./AU_predictors/AU_SVM_BP4D_static.txt" -fx 2000 -fy 2000 -rigid -asvid -simscale 0.7 -simsize 112 '];

        for f2=1:numel(bp4d_2_dirs)
            f2_dir = bp4d_2_dirs(f2).name;
            if(isdir([bp4d_loc, bp4d_dirs(f1).name]))
                
                curr_vid = [bp4d_loc, f1_dir, '/', f2_dir, '/'];

                name = [f1_dir '_' f2_dir];
                output_file = [out_loc name '/'];

                output_aus = [out_loc name '.au.txt'];
                
                command = cat(2, command, [' -fdir "' curr_vid '" -oaus "' output_aus '"']);
            end
        end
        
        dos(command);
    end
end

chdir(oldDir)