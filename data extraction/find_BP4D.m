if(exist('C:\tadas\face_datasets\fera_2015\bp4d\AUCoding/', 'file'))
    BP4D_dir = 'C:\tadas\face_datasets\fera_2015\bp4d\AUCoding/';   
elseif(exist('E:\datasets\FERA_2015\BP4D\AUCoding/', 'file'))
    BP4D_dir = 'E:\datasets\FERA_2015\BP4D\AUCoding/';   
else
    fprintf('BP4D location not found (or not defined)\n'); 
end

hog_data_dir = [BP4D_dir, '../processed_data'];