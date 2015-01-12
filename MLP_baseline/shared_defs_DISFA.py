

def shared_defs():
    all_aus = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
    
    import os.path
    
    # load all of the data together (for efficiency)
    # it will be split up accordingly at later stages
    if os.path.exists('F:/datasets/DISFA/'):
        DISFA_dir = 'F:/datasets/DISFA/'
    elif os.path.exists('D:/Databases/DISFA/'):
        DISFA_dir = 'D:/Databases/DISFA/'
    elif os.path.exists('Z:/datasets/DISFA/'):
        DISFA_dir = 'Z:/datasets/DISFA/'
    elif os.path.exists('E:/datasets/DISFA/'):
        DISFA_dir = 'E:/datasets/DISFA/'
    elif os.path.exists('C:/tadas/DISFA/'):
        DISFA_dir = 'C:/tadas/DISFA/'
    elif os.path.exists('D:/datasets/face_datasets/DISFA/'):
        DISFA_dir = 'D:/datasets/face_datasets/DISFA/'
    else:
        DISFA_dir = ''
        print 'DISFA location not found';
    
    hog_data_dir = DISFA_dir + '/hog_aligned_rigid/';

    users = ['SN001', 'SN002', 'SN003', 'SN004', 'SN005', 'SN006', 'SN007', 'SN008', 'SN009', 'SN010', 'SN011', 'SN012',
             'SN016', 'SN017', 'SN018', 'SN021', 'SN023', 'SN024', 'SN025', 'SN026', 'SN027', 'SN028', 'SN029', 'SN030',
             'SN031', 'SN032', 'SN013']

    return (all_aus, users, DISFA_dir, hog_data_dir);