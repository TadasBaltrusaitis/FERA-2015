if(exist('E:\datasets\FERA_2015\semaine/SEMAINE-Sessions/', 'file'))
    SEMAINE_dir = 'E:\datasets\FERA_2015\semaine/SEMAINE-Sessions/';   
elseif(exist('I:\datasets\FERA_2015\Semaine\SEMAINE-Sessions/', 'file'))
    SEMAINE_dir = 'I:\datasets\FERA_2015\Semaine\SEMAINE-Sessions/';   
elseif(exist('C:\tadas\face_datasets\fera_2015\semaine/SEMAINE-Sessions/', 'file'))
    SEMAINE_dir = 'C:\tadas\face_datasets\fera_2015\semaine/SEMAINE-Sessions/';   
elseif(exist('D:\datasets\face_datasets\fera_2015\semaine\SEMAINE-Sessions/', 'file'))
    SEMAINE_dir = 'D:\datasets\face_datasets\fera_2015\semaine\SEMAINE-Sessions/';
else
    fprintf('DISFA location not found (or not defined)\n'); 
end

if(exist('SEMAINE_dir', 'var'))
    hog_data_dir = [SEMAINE_dir, '../processed_data/'];
end