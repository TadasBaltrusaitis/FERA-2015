# this is data defined across the experiments (to make sure all of them have same user conventions)

# Defining which AU's we are extracting (all corrs above 0.5)
def shared_defs():
    all_aus = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23];
    
    import os.path
    
    # load all of the data together (for efficiency)
    # it will be split up accordingly at later stages
    if(os.path.exists('C:/tadas/face_datasets/fera_2015/bp4d/AUCoding/')):
        BP4D_dir = 'C:/tadas/face_datasets/fera_2015/bp4d/AUCoding/';   
    elif(os.path.exists('E:/datasets/FERA_2015/BP4D/AUCoding/')):
        BP4D_dir = 'E:/datasets/FERA_2015/BP4D/AUCoding/';   
    elif(os.path.exists('D:/datasets/face_datasets/fera_2015/bp4d/AUCoding/')):
        BP4D_dir = 'D:/datasets/face_datasets/fera_2015/bp4d/AUCoding/';   
    elif(os.path.exists('I:/datasets/FERA_2015/BP4D/AUCoding/')):
        BP4D_dir = 'I:/datasets/FERA_2015/BP4D/AUCoding/';
    else:
        BP4D_dir = '';
        print 'BP4D location not found'; 
    
    hog_data_dir = BP4D_dir + '../processed_data/';
    
    # train_recs = {'rec1', 'rec12', 'rec14', 'rec19', 'rec23', 'rec25', 'rec37', 'rec39', 'rec43', 'rec45', 'rec48', 'rec50', 'rec52', 'rec54', 'rec56', 'rec60'};
    # devel_recs = {'rec9', 'rec13', 'rec15', 'rec20', 'rec24', 'rec26', 'rec38', 'rec42', 'rec44', 'rec46', 'rec49', 'rec51', 'rec53', 'rec55', 'rec58'};
    
    # Issues with the following recordings 9, 50, 56, so they are omitted for
    # now
    train_recs = ['F001', 'F003', 'F005', 'F007', 'F009', 'F011', 'F013', 'F015', 'F017', 'F019', 'F021', 'F023', 'M001', 'M003', 'M005', 'M007', 'M009', 'M011', 'M013', 'M015', 'M017'];
    devel_recs = ['F002', 'F004', 'F006', 'F008', 'F010', 'F012', 'F014', 'F016', 'F018', 'F020', 'F022', 'M002', 'M004', 'M006', 'M008', 'M010', 'M012', 'M014', 'M016', 'M018'];
    
    return (all_aus, train_recs, devel_recs, BP4D_dir, hog_data_dir);