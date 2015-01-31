clear

addpath('../data extraction/');
find_BP4D;

bp4d_loc = [BP4D_dir '../BP4D/BP4D-training/'];

out_loc = './txt_files/';

if(~exist(out_loc, 'dir'))
    mkdir(out_loc);
end

% Go two levels deep
bp4d_dirs = devel_recs;
        
for f1=1:numel(bp4d_dirs)

    if(isdir([bp4d_loc, bp4d_dirs{f1}]))
        
        bp4d_2_dirs = dir([bp4d_loc, bp4d_dirs{f1}]);
        bp4d_2_dirs = bp4d_2_dirs(3:end);
        
        f1_dir = bp4d_dirs{f1};

        for f2=1:numel(bp4d_2_dirs)
            f2_dir = bp4d_2_dirs(f2).name;
            if(isdir([bp4d_loc, bp4d_dirs{f1}]))
                
                curr_vid_dir = [bp4d_loc, f1_dir, '/', f2_dir, '/'];

                all_files = dir([curr_vid_dir, '*.jpg']);
                
                f = fopen([out_loc, f1_dir, '_', f2_dir, '.txt'], 'w');
                for i=1:numel(all_files)
                    fprintf(f, '%s/%s/%s/%s\n', bp4d_loc, f1_dir, f2_dir, all_files(i).name);
                end
                fclose(f);
            end
        end

    end
end