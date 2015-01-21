clear

addpath('../data extraction/');
find_BP4D;

bp4d_loc = [BP4D_dir '../BP4D/BP4D-training/'];

out_loc = './out_bp4d_static_intensity/';

if(~exist(out_loc, 'dir'))
    mkdir(out_loc);
end

features_exe = '"../C++ models/Release/AUPrediction.exe"';

% Go two levels deep
bp4d_dirs = devel_recs;

in_txt = './txt_files/';

for f1=1:numel(bp4d_dirs)

    if(isdir([bp4d_loc, bp4d_dirs{f1}]))
        
        bp4d_2_dirs = dir([bp4d_loc, bp4d_dirs{f1}]);
        bp4d_2_dirs = bp4d_2_dirs(3:end);
        
        f1_dir = bp4d_dirs{f1};

        for f2=1:numel(bp4d_2_dirs)

            command = [features_exe, ' -scaling 0.4 -auloc "./AU_predictors/AU_SVR_BP4D_static.txt" -fx 2000 -fy 2000 -rigid -asvid -simscale 0.7 -simsize 112 '];

            f2_dir = bp4d_2_dirs(f2).name;
            if(isdir([bp4d_loc, bp4d_dirs{f1}]))
                
                curr_txt_file = [in_txt, f1_dir, '_', f2_dir, '.txt'];

                name = [f1_dir '_' f2_dir];
                output_file = [out_loc name '/'];

                output_aus = [out_loc name '.au.txt'];
                
                command = cat(2, command, [' -ftxt "' curr_txt_file '" -oaus "' output_aus '"']);
            end
            
            dos(command);
        end
        
    end
end