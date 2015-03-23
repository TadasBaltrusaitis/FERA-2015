% Biwi dataset experiment

features_exe = '"..\C++ models\Release\FeatureExtraction.exe"';

if(exist('D:/Databases/DISFA/', 'dir'))
    DISFA_loc = 'D:/Databases/DISFA/';    
elseif(exist('E:/datasets/DISFA/', 'dir'))
    DISFA_loc = 'E:/datasets/DISFA/';       
elseif(exist('C:/tadas/DISFA', 'dir'))
    DISFA_loc = 'C:/tadas/DISFA/';
elseif(exist('D:\datasets\face_datasets\DISFA/', 'dir'))
    DISFA_loc = 'D:\datasets\face_datasets\DISFA/';    
else
   fprintf('DISFA not found\n'); 
end

output = [DISFA_loc, '/aligned_rigid/'];
output_hog_root = [DISFA_loc '/hog_aligned_rigid/'];
output_params_root = [DISFA_loc '/clm_params/'];

DISFA_loc_1 = [DISFA_loc, 'Videos_LeftCamera/'];
DISFA_loc_2 = [DISFA_loc, 'Video_RightCamera/'];

if(~exist(output, 'dir'))
    mkdir(output);
end

if(~exist(output_hog_root, 'dir'))
    mkdir(output_hog_root);
end

if(~exist(output_params_root, 'dir'))
    mkdir(output_params_root);
end

disfa_loc_1_files = dir([DISFA_loc_1, '/*.avi']);
disfa_loc_2_files = dir([DISFA_loc_2, '/*.avi']);

%%

tic;

parfor i=1:numel(disfa_loc_1_files)
           
    command = features_exe;
               
    input_file = [DISFA_loc_1 disfa_loc_1_files(i).name];
        
    [~,name,~] = fileparts(disfa_loc_1_files(i).name);
    output_file = [output name '/'];

    output_hog = [output_hog_root name '.hog'];
    output_params = [output_params_root '/' name '.txt'];
        
    command = cat(2, command, [' -rigid -f "' input_file '" -simalign "' output_file  '" -simscale 0.7 -simsize 112']);
    command = cat(2, command, [' -hogalign "' output_hog '"' ]);
    command = cat(2, command, [' -oparams "' output_params '"']);

    dos(command);
end

%%
parfor i=1:numel(disfa_loc_2_files)
           
    command = features_exe;
               
    input_file = [DISFA_loc_2 disfa_loc_2_files(i).name];
        
    [~,name,~] = fileparts(disfa_loc_2_files(i).name);
    output_file = [output name '/'];
    
    output_hog = [output_hog_root name '.hog'];
        
    output_params = [output_params_root '/' name '.txt'];
        
    command = cat(2, command, [' -rigid -f "' input_file '" -simalign "' output_file  '" -simscale 0.7 -simsize 112']);  
    command = cat(2, command, [' -hogalign "' output_hog '"']);
    command = cat(2, command, [' -oparams "' output_params '"']);
    
    dos(command);
end

timeTaken = toc;