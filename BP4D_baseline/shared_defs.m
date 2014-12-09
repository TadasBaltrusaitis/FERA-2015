% this is data defined across the experiments (to make sure all of them have same user conventions)

% Defining which AU's we are extracting (all corrs above 0.5)
all_aus = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23];
aus = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23];

% load all of the data together (for efficiency)
% it will be split up accordingly at later stages
if(exist('C:\tadas\face_datasets\fera_2015\bp4d\AUCoding/', 'file'))
    BP4D_dir = 'C:\tadas\face_datasets\fera_2015\bp4d\AUCoding/';   
else    
    fprintf('BP4D location not found (or not defined)\n'); 
end

if(exist('BP4D_dir', 'var'))
    hog_data_dir = [BP4D_dir, '../processed_data/'];
end

train_recs = {'F001', 'F003', 'F005', 'F007', 'F009', 'F011', 'F013', 'F015', 'F017', 'F019', 'F021', 'F023', 'M001', 'M003', 'M005', 'M007', 'M009', 'M011', 'M013', 'M015' 'M017'};
devel_recs = {'F002', 'F004', 'F006', 'F008', 'F010', 'F012', 'F014', 'F016', 'F018', 'F020', 'F022', 'M002', 'M004', 'M006', 'M008', 'M010', 'M012', 'M014', 'M016', 'M018'};

