clear

addpath('../data extraction/');
find_BP4D;

bp4d_loc = [BP4D_dir '../BP4D-training/'];

out_loc = './out_bp4d_all_best/';

if(~exist(out_loc, 'dir'))
    mkdir(out_loc);
end

features_exe = '"../C++ models/script_au_pred.bat"';

% Go two levels deep
bp4d_dirs = devel_recs;
        
in_txt = './txt_files/';

parfor f1=1:numel(bp4d_dirs)

    if(isdir([bp4d_loc, bp4d_dirs{f1}]))
        
        bp4d_2_dirs = dir([bp4d_loc, bp4d_dirs{f1}]);
        bp4d_2_dirs = bp4d_2_dirs(3:end);
        
        f1_dir = bp4d_dirs{f1};

        for f2=1:numel(bp4d_2_dirs)

            command = features_exe;

            f2_dir = bp4d_2_dirs(f2).name;
            if(isdir([bp4d_loc, bp4d_dirs{f1}]))
                
                curr_txt_file = ['"' in_txt, f1_dir, '_', f2_dir, '.txt"'];

                database = 'BP4D';
                
                name = [f1_dir '_' f2_dir];
                output_file = [out_loc name '/'];

                output_aus_class = ['"' out_loc name '.au.class.txt"'];
                output_aus_reg = ['"' out_loc name '.au.reg.txt"'];
                output_aus_reg_seg = ['"' out_loc name '.au.reg.seg.txt"'];
                
                command = cat(2, command, [' ' curr_txt_file ' ' database ' empty ' output_aus_class ' ' output_aus_reg ' ' output_aus_reg_seg]);
            end
            
            dos(command);
        end
        
    end
end