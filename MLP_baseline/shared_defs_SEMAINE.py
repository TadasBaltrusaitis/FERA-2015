# this is data defined across the experiments (to make sure all of them have same user conventions)

# Defining which AU's we are extracting (all corrs above 0.5)
def shared_defs():
    all_aus = [2, 12, 17, 25, 28, 45]

    import os.path

    # load all of the data together (for efficiency)
    # it will be split up accordingly at later stages
    if os.path.exists('E:/datasets/FERA_2015/semaine/SEMAINE-Sessions/'):
        SEMAINE_dir = 'E:/datasets/FERA_2015/semaine/SEMAINE-Sessions/'
    elif os.path.exists('I:/datasets/FERA_2015/Semaine/SEMAINE-Sessions/'):
        SEMAINE_dir = 'I:/datasets/FERA_2015/Semaine/SEMAINE-Sessions/'
    elif os.path.exists('C:/tadas/face_datasets/fera_2015/semaine/SEMAINE-Sessions/'):
        SEMAINE_dir = 'C:/tadas/face_datasets/fera_2015\semaine/SEMAINE-Sessions/'
    elif os.path.exists('D:/datasets/face_datasets/fera_2015/semaine/SEMAINE-Sessions/'):
        SEMAINE_dir = 'D:/datasets/face_datasets/fera_2015/semaine/SEMAINE-Sessions/'
    else:
        SEMAINE_dir = ''
        print 'DISFA location not found (or not defined)'

    hog_data_dir = SEMAINE_dir + '../processed_data/'

    train_recs = ['rec1', 'rec12', 'rec14', 'rec19', 'rec23', 'rec25', 'rec37', 'rec39', 'rec43', 'rec45', 'rec48',
                  'rec50', 'rec52', 'rec54', 'rec56', 'rec60']
    devel_recs = ['rec9', 'rec13', 'rec15', 'rec20', 'rec24', 'rec26', 'rec38', 'rec42', 'rec44', 'rec46', 'rec49',
                  'rec51', 'rec53', 'rec55', 'rec58']

    return (all_aus, train_recs, devel_recs, SEMAINE_dir, hog_data_dir)