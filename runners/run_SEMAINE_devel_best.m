clear

addpath(genpath('../data extraction/'));
find_SEMAINE;

out_loc = './out_SEMAINE_best/';

if(~exist(out_loc, 'dir'))
    mkdir(out_loc);
end

features_exe = '"../C++ models/Release/AUPrediction.exe"';

% Go two levels deep
aus_SEMAINE = [2];

[ labels, valid_ids, vid_ids  ] = extract_SEMAINE_labels(SEMAINE_dir, devel_recs, aus_SEMAINE);

parfor f1=1:numel(devel_recs)


    if(isdir([SEMAINE_dir, devel_recs{f1}]))
        
        vid_files = dir([SEMAINE_dir, devel_recs{f1}, '/*.avi']);

        f1_dir = devel_recs{f1};
        
        for v=1:numel(vid_files)
    
            command = [features_exe, ' -auloc "./AU_predictors/AU_SVM_SEMAINE_best.txt" -fx 800 -fy 800 -rigid -asvid -simscale 0.7 -simsize 112 '];
    
            curr_vid = [SEMAINE_dir, f1_dir, '/', vid_files(v).name];

            name = f1_dir;
            output_aus = [out_loc name '.au.txt'];

            command = cat(2, command, [' -f "' curr_vid '" -oausclass "' output_aus '" -ef ' num2str(vid_ids(f1,2)) ' -bf ' num2str(vid_ids(f1,1))]);            
            dos(command);

        end
    end
end