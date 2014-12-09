% this is data defined across the experiments (to make sure all of them have same user conventions)

% Defining which AU's we are extracting (all corrs above 0.5)
all_aus = [2 12 17 25 28 45];
aus = [2 12 17 25 28 45];

% load all of the data together (for efficiency)
% it will be split up accordingly at later stages
if(exist('E:\datasets\FERA_2015\semaine/SEMAINE-Sessions/', 'file'))
    SEMAINE_dir = 'E:\datasets\FERA_2015\semaine/SEMAINE-Sessions/';   
elseif(exist('I:\datasets\FERA_2015\Semaine\SEMAINE-Sessions/', 'file'))
    SEMAINE_dir = 'I:\datasets\FERA_2015\Semaine\SEMAINE-Sessions/';   
elseif(exist('C:\tadas\face_datasets\fera_2015\semaine/SEMAINE-Sessions/', 'file'))
    SEMAINE_dir = 'C:\tadas\face_datasets\fera_2015\semaine/SEMAINE-Sessions/';   
else    
    fprintf('DISFA location not found (or not defined)\n'); 
end

if(exist('SEMAINE_dir', 'var'))
    hog_data_dir = [SEMAINE_dir, '../processed_data/'];
end

% train_recs = {'rec1', 'rec12', 'rec14', 'rec19', 'rec23', 'rec25', 'rec37', 'rec39', 'rec43', 'rec45', 'rec48', 'rec50', 'rec52', 'rec54', 'rec56', 'rec60'};
% devel_recs = {'rec9', 'rec13', 'rec15', 'rec20', 'rec24', 'rec26', 'rec38', 'rec42', 'rec44', 'rec46', 'rec49', 'rec51', 'rec53', 'rec55', 'rec58'};

% Issues with the following recordings 9, 50, 56, so they are omitted for
% now
train_recs = {'rec1', 'rec12', 'rec14', 'rec19', 'rec23', 'rec25', 'rec37', 'rec39', 'rec43', 'rec45', 'rec48', 'rec52', 'rec54', 'rec60'};
devel_recs = {'rec13', 'rec15', 'rec20', 'rec24', 'rec26', 'rec38', 'rec42', 'rec44', 'rec46', 'rec49', 'rec51', 'rec53', 'rec55', 'rec58'};


