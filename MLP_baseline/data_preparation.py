import numpy as np
import scipy.io

def extract_BP4D_labels_intensity(BP4D_dir, recs, aus):

    import glob
    import os.path

    files_all = glob.glob('%s/AU%02d/*.csv' % (BP4D_dir, aus[0]))
    num_files = len(files_all)

    labels_all = []
    valid_ids_all = []
    # This should be ndarray
    vid_ids = np.zeros((num_files, 2))
    filenames_all = []

    file_id = 0

    for r in range(len(recs)):

        files_root = '%s/AU%02d/' % (BP4D_dir, aus[0])
        files_all = glob.glob(files_root + str(recs[r]) + '*.csv')

        for f in range(len(files_all)):
            for au in aus:

                # Need to find relevant files for the relevant user and for the
                # relevant AU
                files_root = '%s/AU%02d/' % (BP4D_dir, au)
                files_all = glob.glob(files_root  + str(recs[r]) + '*.csv')

                file = files_all[f]

                _, filename = os.path.split(file)

                # import annotations for one session
                intensities = np.genfromtxt(file, delimiter=',')

                # get all frame numbers
                frame_nums = intensities[:,0]

                codes = intensities[:,1:2]

                # Finding the invalid regions
                valid = codes != 9

                vid_ids[file_id,0] = frame_nums[0]
                vid_ids[file_id,1] = frame_nums[-1]

                if au == aus[0]:
                    valid_ids = valid
                    labels = codes
                else:
                    valid_ids = np.logical_and(valid_ids, valid)
                    labels = np.concatenate((labels, codes), axis=1)

            labels_all.append(labels)
            valid_ids_all.append(valid_ids)
            filenames_all += [filename[0:7]]

            file_id = file_id + 1

    vid_ids = vid_ids[0:file_id, :]

    return labels_all, valid_ids_all, vid_ids, filenames_all

def extract_DISFA_labels(input_folders, aus):

    from numpy import genfromtxt
    from os import path

    labels_all = None
    vid_ids = None

    for input_folder in input_folders:

        tail, _ = path.split(input_folder)
        _, name = path.split(tail)

        labels_curr_fold = None

        for au in aus:

            in_file = '%s_au%d.txt' % (input_folder, au)

            labels_curr = genfromtxt(in_file, dtype=int, delimiter=',')
            labels_curr = labels_curr[:, 1:2]

            if labels_curr_fold is None:
                labels_curr_fold = labels_curr
            else:
                labels_curr_fold = np.concatenate((labels_curr_fold, labels_curr), axis=1)

        if labels_all is None:
            labels_all = labels_curr_fold
            vid_ids = [name] * labels_curr_fold.shape[0]
        else:
            labels_all = np.concatenate((labels_all, labels_curr_fold), axis=0)
            vid_ids += [name] * labels_curr_fold.shape[0]

    return labels_all, vid_ids

def extract_SEMAINE_labels(SEMAINE_label_dir, recs, aus):

    labels = []
    valid_ids = []
    # This should be ndarray
    vid_ids = np.zeros((len(recs), 2))

    ind = 0
    for rec in recs:
        labels_rec_all = None
        valid_ids_rec_all = None
        for au in aus:

            name = SEMAINE_label_dir + rec + '_AU' + str(au) + '.mat'

            dim_reds = scipy.io.loadmat(name)

            labels_rec = dim_reds['labels_rec']
            valid_ids_rec = dim_reds['valid_ids_rec']
            vid_ids_rec = dim_reds['vid_ids_rec'][0]

            if labels_rec_all is None:

                labels_rec_all = labels_rec
                valid_ids_rec_all = valid_ids_rec

            else:

                labels_rec_all = np.concatenate((labels_rec_all, labels_rec), axis=1)
                valid_ids_rec_all = np.logical_and(valid_ids_rec_all, vid_ids_rec)

            vid_ids[ind, :] = [vid_ids_rec[0], vid_ids_rec[1]]

        labels.append(labels_rec_all)
        valid_ids.append(valid_ids_rec_all)
        ind += 1

    return labels, valid_ids, vid_ids


def extract_BP4D_labels(BP4D_dir, recs, aus):
    
    import glob
    import os
    import numpy as np
    
    aus_BP4D_all = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23]
    
    inds_to_use = []
    
    for au in aus:    
        inds_to_use = inds_to_use + [aus_BP4D_all.index(au)]
        
    num_files = len(glob.glob(BP4D_dir + '/*.csv'))
    
    labels = []
    valid_ids = []
    # This should be ndarray
    vid_ids = np.zeros((num_files, 2))
    filenames = []
    
    file_id = 0
    
    for i in range(len(recs)):

        csvs = glob.glob(BP4D_dir + '/' + recs[i] + '*.csv')
    
        for f in range(len(csvs)):
            
            filepath = csvs[f]

            _, filename_full = os.path.split(csvs[f])
            filename, _ = os.path.splitext(filename_full)
            
            filenames = filenames + [filename]
            OCC = np.genfromtxt(filepath, delimiter=',')

            # get all frame numbers
            frame_nums = OCC[1:, 0]
            codes = OCC[1:, 1:]
            # get codes for all action units
            occlusions = OCC[1:, -1]
            
            codes = codes[:, np.array(aus_BP4D_all) - 1]
            
            # Finding the invalid regions
            valid = occlusions != 1
            
            for s in range(codes.shape[1]):              
                valid = np.logical_and(valid, (codes[:, s] != 9))
                            
            vid_ids[file_id, :] = [frame_nums[0], frame_nums[-1]]

            labels.append(codes[:, inds_to_use])
            valid_ids.append(valid)
            
            file_id += 1
            
    vid_ids = vid_ids[0:file_id, :]

    return labels, valid_ids, vid_ids

def Read_geom_files_DISFA(users, hog_data_dir):

    from numpy import genfromtxt

    geom_data = None

    for i in range(len(users)):

        in_file = hog_data_dir + '../clm_params/LeftVideo' + users[i] + "_comp.txt"
        data_curr = genfromtxt(in_file, dtype=float, delimiter=' ')
        data_curr = data_curr[:, 14::2]

        if geom_data is None:
            geom_data = data_curr
        else:
            geom_data = np.concatenate((geom_data, data_curr), axis=0)

    return geom_data

def Read_geom_files_DISFA_dynamic(users, hog_data_dir):

    from numpy import genfromtxt

    geom_data = None

    for i in range(len(users)):

        in_file = hog_data_dir + '../clm_params/LeftVideo' + users[i] + "_comp.txt"
        data_curr = genfromtxt(in_file, dtype=float, delimiter=' ')
        data_curr = data_curr[:, 14::2]

        data_curr = data_curr - np.median(data_curr, axis=0)

        if geom_data is None:
            geom_data = data_curr
        else:
            geom_data = np.concatenate((geom_data, data_curr), axis=0)

    return geom_data

def Read_HOG_files_DISFA(users, hog_data_dir):

    import struct

    vid_id = []
    valid_inds = []

    feats_filled = 0
    hog_data = np.array((0, 0))

    for i in range(len(users)):

        hog_file = hog_data_dir + 'LeftVideo' + users[i] + '_comp.hog'

        f = open(hog_file, 'rb')

        curr_data = []
        curr_ind = 0
        try:
            while True:

                if curr_ind == 0:

                    a = f.read(4)

                    if not a:
                        break

                    num_cols = struct.unpack('i', a)[0]

                    num_rows = struct.unpack('i', f.read(4))[0]
                    num_chan = struct.unpack('i', f.read(4))[0]

                    # preallocate some space
                    if curr_ind == 0:
                        curr_data = np.zeros((5000, 1 + num_rows * num_cols * num_chan))
                        num_feats =  1 + num_rows * num_cols * num_chan

                    # Add more spce to the buffer
                    if curr_ind >= curr_data.shape[0]:
                        curr_data = np.concatenate(curr_data, np.zeros((5000, 1 + num_rows * num_cols * num_chan)))

                    feature_vec = np.fromfile(f, dtype='float32', count = 1 + num_rows * num_cols * num_chan)
                    curr_data[curr_ind, :] = feature_vec;

                    curr_ind += 1

                else:

                    # Reading in batches of 5000

                    feature_vec = np.fromfile(f, dtype='float32', count=(3 + num_feats) * 5000)
                    if(feature_vec.shape[0]==0):
                        break

                    feature_vec.shape = (feature_vec.shape[0]/(3+num_feats), 3 + num_feats)

                    feature_vec = feature_vec[:, 3:]

                    num_rows_read = feature_vec.shape[0]

                    curr_data[curr_ind:curr_ind+num_rows_read,:] = feature_vec

                    curr_ind = curr_ind + feature_vec.shape[0]
        finally:
            f.close()

        curr_data = curr_data[0:curr_ind, :]
        vid_id_curr = [users[i]] * curr_ind

        vid_id.append(vid_id_curr)

        # Assume same number of frames per video
        if i == 0:
            hog_data = np.zeros((curr_ind * len(users), num_feats))

        if hog_data.shape[0] < feats_filled+curr_ind:
            hog_data = np.concatenate(hog_data, np.zeros(hog_data.shape[0], num_feats))

        hog_data[feats_filled:feats_filled+curr_ind,:] = curr_data

        feats_filled = feats_filled + curr_ind

    if hog_data.shape[0] != 0:
        valid_inds = hog_data[0:feats_filled, 0]
        hog_data = hog_data[0:feats_filled, 1:]

    return hog_data, valid_inds, vid_id


def Read_HOG_files_DISFA_dynamic(users, hog_data_dir):

    import struct

    vid_id = []
    valid_inds = []

    feats_filled = 0
    hog_data = np.array((0, 0))

    for i in range(len(users)):

        hog_file = hog_data_dir + 'LeftVideo' + users[i] + '_comp.hog'

        f = open(hog_file, 'rb')

        curr_data = []
        curr_ind = 0
        try:
            while True:

                if curr_ind == 0:

                    a = f.read(4)

                    if not a:
                        break

                    num_cols = struct.unpack('i', a)[0]

                    num_rows = struct.unpack('i', f.read(4))[0]
                    num_chan = struct.unpack('i', f.read(4))[0]

                    # preallocate some space
                    if curr_ind == 0:
                        curr_data = np.zeros((5000, 1 + num_rows * num_cols * num_chan))
                        num_feats =  1 + num_rows * num_cols * num_chan

                    # Add more spce to the buffer
                    if curr_ind >= curr_data.shape[0]:
                        curr_data = np.concatenate(curr_data, np.zeros((5000, 1 + num_rows * num_cols * num_chan)))

                    feature_vec = np.fromfile(f, dtype='float32', count = 1 + num_rows * num_cols * num_chan)
                    curr_data[curr_ind, :] = feature_vec;

                    curr_ind += 1

                else:

                    # Reading in batches of 5000

                    feature_vec = np.fromfile(f, dtype='float32', count=(3 + num_feats) * 5000)
                    if(feature_vec.shape[0]==0):
                        break

                    feature_vec.shape = (feature_vec.shape[0]/(3+num_feats), 3 + num_feats)

                    feature_vec = feature_vec[:, 3:]

                    num_rows_read = feature_vec.shape[0]

                    curr_data[curr_ind:curr_ind+num_rows_read,:] = feature_vec

                    curr_ind = curr_ind + feature_vec.shape[0]
        finally:
            f.close()

        curr_data = curr_data[0:curr_ind, :]

        curr_data[:, 1:] = curr_data[:, 1:] - np.median(curr_data[:, 1:], axis=0)

        vid_id_curr = [users[i]] * curr_ind

        vid_id.append(vid_id_curr)

        # Assume same number of frames per video
        if i == 0:
            hog_data = np.zeros((curr_ind * len(users), num_feats))

        if hog_data.shape[0] < feats_filled+curr_ind:
            hog_data = np.concatenate(hog_data, np.zeros(hog_data.shape[0], num_feats))

        hog_data[feats_filled:feats_filled+curr_ind,:] = curr_data

        feats_filled = feats_filled + curr_ind

    if hog_data.shape[0] != 0:
        valid_inds = hog_data[0:feats_filled, 0]
        hog_data = hog_data[0:feats_filled, 1:]

    return hog_data, valid_inds, vid_id

def Read_geom_files_SEMAINE(users, hog_data_dir, vid_ids):

    from numpy import genfromtxt

    geom_data = None

    for i in range(len(users)):

        in_file = hog_data_dir + '/' + users[i] + ".params.txt"
        data_curr = genfromtxt(in_file, dtype=float, delimiter=' ')
        data_curr = data_curr[vid_ids[i,0]:vid_ids[i,1], 14::2]

        if geom_data is None:
            geom_data = data_curr
        else:
            geom_data = np.concatenate((geom_data, data_curr), axis=0)

    return geom_data

def Read_geom_files_SEMAINE_dynamic(users, hog_data_dir, vid_ids):

    from numpy import genfromtxt

    geom_data = None

    for i in range(len(users)):

        in_file = hog_data_dir + '/' + users[i] + ".params.txt"
        data_curr = genfromtxt(in_file, dtype=float, delimiter=' ')
        data_curr = data_curr[vid_ids[i,0]:vid_ids[i,1], 14::2]
        data_curr = data_curr - np.median(data_curr, axis=0)

        if geom_data is None:
            geom_data = data_curr
        else:
            geom_data = np.concatenate((geom_data, data_curr), axis=0)

    return geom_data


def Read_geom_files_BP4D(users, hog_data_dir):

    import glob
    from numpy import genfromtxt

    geom_data = None

    files = []

    for i in range(len(users)):

        geom_files = glob.glob(hog_data_dir + users[i] + '*.params.txt')

        for h in range(len(geom_files)):
            in_file = geom_files[h]

            files += [in_file]

            data_curr = genfromtxt(in_file, dtype=float, delimiter=' ')
            data_curr = data_curr[:, 14::2]

            if geom_data is None:
                geom_data = data_curr
            else:
                geom_data = np.concatenate((geom_data, data_curr), axis=0)

    return geom_data, files

def Read_geom_files_BP4D_dynamic(users, hog_data_dir):

    import glob
    from numpy import genfromtxt

    geom_data = None

    files = []

    for i in range(len(users)):

        geom_files = glob.glob(hog_data_dir + users[i] + '*.params.txt')

        geom_data_c = None

        for h in range(len(geom_files)):
            in_file = geom_files[h]

            files += [in_file]

            data_curr = genfromtxt(in_file, dtype=float, delimiter=' ')
            data_curr = data_curr[:, 14::2]

            if geom_data_c is None:
                geom_data_c = data_curr
            else:
                geom_data_c = np.concatenate((geom_data_c, data_curr), axis=0)

        geom_data_c = geom_data_c - np.median(geom_data_c, axis=0)

        if geom_data is None:
            geom_data = geom_data_c
        else:
            geom_data = np.concatenate((geom_data, geom_data_c), axis=0)

    return geom_data, files


def Read_HOG_files_BP4D(users, hog_data_dir):
    
    import glob
    import struct
    
    vid_id = []
    valid_inds = []
    
    feats_filled = 0
    hog_data = np.array((0, 0))
    
    for i in range(len(users)):
        
        hog_files = glob.glob(hog_data_dir + users[i] + '*.hog')
        
        for h in range(len(hog_files)):

            hog_file = hog_files[h]
            
            f = open(hog_file, 'rb')

            curr_data = []
            curr_ind = 0
            try:
                while True:

                    if curr_ind == 0:
                        
                        a = f.read(4)

                        if(not a):
                            break

                        num_cols = struct.unpack('i', a)[0]
    
                        num_rows = struct.unpack('i', f.read(4))[0]
                        num_chan = struct.unpack('i', f.read(4))[0]
    
                        # preallocate some space
                        if(curr_ind == 0):
                            curr_data = np.zeros((1000, 1 + num_rows * num_cols * num_chan))
                            num_feats =  1 + num_rows * num_cols * num_chan
                        
                        # Add more spce to the buffer
                        if curr_ind >= curr_data.shape[0]:
                            curr_data = np.concatenate(curr_data, np.zeros((1000, 1 + num_rows * num_cols * num_chan)))

                        feature_vec = np.fromfile(f, dtype='float32', count = 1 + num_rows * num_cols * num_chan)
                        curr_data[curr_ind, :] = feature_vec;

                        curr_ind += 1
                            
                    else:
    
                        # Reading in batches of 5000
    
                        feature_vec = np.fromfile(f, dtype='float32', count=(3 + num_feats) * 5000);
                        if(feature_vec.shape[0]==0):
                            break;

                        feature_vec.shape = (feature_vec.shape[0]/(3+num_feats), 3 + num_feats)
                                                        
                        feature_vec = feature_vec[:,3:]

                        num_rows_read = feature_vec.shape[0]
    
                        curr_data[curr_ind:curr_ind+num_rows_read,:] = feature_vec
    
                        curr_ind = curr_ind + feature_vec.shape[0]
            finally:
                f.close()
                
            curr_data = curr_data[0:curr_ind, :]
            vid_id_curr = [users[i]] * curr_ind

            vid_id.append(vid_id_curr)

            # Assume same number of frames per video
            if i == 0 and h == 0:
                hog_data = np.zeros((curr_ind * len(users) * 8, num_feats))

            if hog_data.shape[0] < feats_filled+curr_ind:
                hog_data = np.concatenate(hog_data, np.zeros(hog_data.shape[0], num_feats))

            hog_data[feats_filled:feats_filled+curr_ind,:] = curr_data

            feats_filled = feats_filled + curr_ind
    
    if hog_data.shape[0] != 0:
        valid_inds = hog_data[0:feats_filled, 0]
        hog_data = hog_data[0:feats_filled, 1:]

    return hog_data, valid_inds, vid_id


def Read_HOG_files_SEMAINE(users, vid_ids, hog_data_dir):

    import struct

    vid_id = []
    valid_inds = []

    hog_data = np.array((0, 0))

    for i in range(len(users)):

        hog_file = hog_data_dir + '/' + users[i] + '.hog'

        f = open(hog_file, 'rb')

        a = f.read(4)

        if not a:
            break

        num_cols = struct.unpack('i', a)[0]

        num_rows = struct.unpack('i', f.read(4))[0]
        num_chan = struct.unpack('i', f.read(4))[0]

        # Read only the relevant bits
        num_feats = num_cols * num_rows * num_chan + 1

        # Skip to the right start element
        f.seek(4*(4+num_rows*num_rows*num_chan)*(vid_ids[i,0]-1), 0)

        vid_len = int(vid_ids[i,1] - vid_ids[i,0])

        feature_vec = np.fromfile(f, dtype='float32', count=(3 + num_feats) * vid_len)

        feature_vec.shape = (feature_vec.shape[0]/(3+num_feats), 3 + num_feats)

        feature_vec = feature_vec[:,3:]

        num_rows_read = feature_vec.shape[0]

        if i == 0:
            hog_data = feature_vec
        else:
            hog_data = np.concatenate((hog_data, feature_vec))

        f.close()

        vid_id_curr = [users[i]] * num_rows_read

        vid_id.append(vid_id_curr)

    if hog_data.shape[0] != 0:
        valid_inds = hog_data[:, 0]
        hog_data = hog_data[:, 1:]

    return hog_data, valid_inds, vid_id


def Read_HOG_files_SEMAINE_dynamic(users, vid_ids, hog_data_dir):

    import struct

    vid_id = []
    valid_inds = []

    hog_data = np.array((0, 0))

    for i in range(len(users)):

        hog_file = hog_data_dir + '/' + users[i] + '.hog'

        f = open(hog_file, 'rb')

        a =  f.read(4)

        if not a:
            break

        num_cols = struct.unpack('i', a)[0]

        num_rows = struct.unpack('i', f.read(4))[0]
        num_chan = struct.unpack('i', f.read(4))[0]

        # Read only the relevant bits
        num_feats = num_cols * num_rows * num_chan + 1

        # Skip to the right start element
        f.seek(4*(4+num_rows*num_rows*num_chan)*(vid_ids[i,0]-1), 0)

        vid_len = int(vid_ids[i,1] - vid_ids[i,0])

        feature_vec = np.fromfile(f, dtype='float32', count=(3 + num_feats) * vid_len)

        feature_vec.shape = (feature_vec.shape[0]/(3+num_feats), 3 + num_feats)

        feature_vec = feature_vec[:,3:]

        num_rows_read = feature_vec.shape[0]

        feature_vec[:, 1:] = feature_vec[:, 1:] - np.median(feature_vec[:, 1:], axis=0)

        if i == 0:
            hog_data = feature_vec
        else:
            hog_data = np.concatenate((hog_data, feature_vec))

        f.close()

        vid_id_curr = [users[i]] * num_rows_read

        vid_id.append(vid_id_curr)

    if hog_data.shape[0] != 0:
        valid_inds = hog_data[:, 0]
        hog_data = hog_data[:, 1:]

    return hog_data, valid_inds, vid_id

# Preparing the SEMAINE data
def Prepare_HOG_AU_data_generic_SEMAINE(train_recs, devel_recs, au, SEMAINE_dir, hog_data_dir, pca_loc, scale=False, geometry=False):

    # First extracting the labels
    SEMAINE_label_dir = '../SEMAINE_baseline/training_labels/'

    [labels_train, valid_ids_train, vid_ids_train] = extract_SEMAINE_labels(SEMAINE_label_dir, train_recs, au)

    train_geom_data = Read_geom_files_SEMAINE(train_recs, hog_data_dir + '/train/', vid_ids_train)

    # Reading in the HOG data (of only relevant frames)
    [train_appearance_data, valid_ids_train_hog, vid_ids_train_string] = \
        Read_HOG_files_SEMAINE(train_recs, vid_ids_train, hog_data_dir + '/train/')


    # Subsample the data to make training quicker
    labels_train = np.concatenate(labels_train)
    valid_ids_train = np.concatenate(valid_ids_train).astype('bool')

    if len(au) == 1:
        labels_train = labels_train[:, 0]

    valid_ids_train = valid_ids_train[:, 0]

    reduced_inds = np.ones((labels_train.shape[0], ), dtype='bool')

    # only remove the data if single au used
    if len(au) == 1:
        # Remove two thirds of negative examples (to balance the training data a bit)
        inds_train = np.array(range(labels_train.shape[0]))
        neg_samples = inds_train[labels_train == 0]

        to_rem = neg_samples[np.round(np.linspace(0, neg_samples.shape[0]-1, neg_samples.shape[0]/1.5).astype('int32'))]
        reduced_inds[to_rem] = False

    # also remove invalid ids based on CLM failing or AU not being labelled
    reduced_inds[valid_ids_train == False] = False
    reduced_inds[valid_ids_train_hog == False] = False

    if len(au) == 1:
        labels_train = labels_train[reduced_inds]
    else:
        labels_train = labels_train[reduced_inds, :]

    train_appearance_data = train_appearance_data[reduced_inds, :]
    train_geom_data = train_geom_data[reduced_inds, :]

    # Extract devel data

    # First extracting the labels
    [labels_devel, valid_ids_devel, vid_ids_devel] = extract_SEMAINE_labels(SEMAINE_label_dir, devel_recs, au)

    # Reading in the HOG data (of only relevant frames)
    [devel_appearance_data, valid_ids_devel_hog, vid_ids_devel_string] = \
        Read_HOG_files_SEMAINE(devel_recs, vid_ids_devel, hog_data_dir + '/devel/')

    devel_geom_data = Read_geom_files_SEMAINE(devel_recs, hog_data_dir +'devel', vid_ids_devel)

    labels_devel = np.concatenate(labels_devel)

    # normalise the data
    dim_reds = scipy.io.loadmat(pca_loc)
    PC = dim_reds['PC']
    means = dim_reds['means_norm']
    scaling = dim_reds['stds_norm']

    # Grab all data for validation as want good params for all the data
    raw_devel = devel_appearance_data
    devel_appearance_data = (devel_appearance_data - means)
    devel_appearance_data = devel_appearance_data/scaling

    train_appearance_data = (train_appearance_data - means)/scaling

    data_train = np.dot(train_appearance_data, PC)
    data_devel = np.dot(devel_appearance_data, PC)

    # TODO ATM scaling is without geom
    if scale:

        # Some extra scaling
        scaling = np.std(data_train, axis=0)

        data_train = data_train / scaling
        data_devel = data_devel / scaling

        PC = PC / scaling

    if geometry:
        data_train = np.concatenate((data_train, train_geom_data), axis=1)
        data_devel = np.concatenate((data_devel, devel_geom_data), axis=1)

    return data_train, labels_train, data_devel, labels_devel, raw_devel, PC, means, scaling

# Preparing the SEMAINE data
def Prepare_HOG_AU_data_generic_SEMAINE_dynamic(train_recs, devel_recs, au, SEMAINE_dir, hog_data_dir, pca_loc, scale=False, geometry=False):

    # First extracting the labels
    SEMAINE_label_dir = '../SEMAINE_baseline/training_labels/'

    [labels_train, valid_ids_train, vid_ids_train] = extract_SEMAINE_labels(SEMAINE_label_dir, train_recs, au)

    train_geom_data = Read_geom_files_SEMAINE_dynamic(train_recs, hog_data_dir + '/train/', vid_ids_train)

    # Reading in the HOG data (of only relevant frames)
    [train_appearance_data, valid_ids_train_hog, vid_ids_train_string] = \
        Read_HOG_files_SEMAINE_dynamic(train_recs, vid_ids_train, hog_data_dir + '/train/')

    # Subsample the data to make training quicker
    labels_train = np.concatenate(labels_train)
    valid_ids_train = np.concatenate(valid_ids_train).astype('bool')

    if len(au) == 1:
        labels_train = labels_train[:, 0]

    valid_ids_train = valid_ids_train[:, 0]

    reduced_inds = np.ones((labels_train.shape[0], ), dtype='bool')

    # only remove the data if single au used
    #if len(au) == 1:
    #    # Remove two thirds of negative examples (to balance the training data a bit)
    #    inds_train = np.array(range(labels_train.shape[0]))
    #    neg_samples = inds_train[labels_train == 0]#

        #to_rem = neg_samples[np.round(np.linspace(0, neg_samples.shape[0]-1, neg_samples.shape[0]/1.5).astype('int32'))]
        #reduced_inds[to_rem] = False

    # also remove invalid ids based on CLM failing or AU not being labelled
    reduced_inds[valid_ids_train == False] = False
    reduced_inds[valid_ids_train_hog == False] = False

    if len(au) == 1:
        labels_train = labels_train[reduced_inds]
    else:
        labels_train = labels_train[reduced_inds, :]

    train_appearance_data = train_appearance_data[reduced_inds, :]
    train_geom_data = train_geom_data[reduced_inds, :]

    # Extract devel data

    # First extracting the labels
    [labels_devel, valid_ids_devel, vid_ids_devel] = extract_SEMAINE_labels(SEMAINE_label_dir, devel_recs, au)

    # Reading in the HOG data (of only relevant frames)
    [devel_appearance_data, valid_ids_devel_hog, vid_ids_devel_string] = \
        Read_HOG_files_SEMAINE_dynamic(devel_recs, vid_ids_devel, hog_data_dir + '/devel/')

    devel_geom_data = Read_geom_files_SEMAINE_dynamic(devel_recs, hog_data_dir +'devel', vid_ids_devel)

    labels_devel = np.concatenate(labels_devel)

    # normalise the data
    dim_reds = scipy.io.loadmat(pca_loc)
    PC = dim_reds['PC']
    means = dim_reds['means_norm']
    scaling = dim_reds['stds_norm']

    # Grab all data for validation as want good params for all the data
    raw_devel = devel_appearance_data
    devel_appearance_data = (devel_appearance_data - means)
    devel_appearance_data = devel_appearance_data/scaling

    train_appearance_data = (train_appearance_data - means)/scaling

    data_train = np.dot(train_appearance_data, PC)
    data_devel = np.dot(devel_appearance_data, PC)

    if scale:

        # Some extra scaling
        scaling = np.std(data_train, axis=0)

        data_train = data_train / scaling
        data_devel = data_devel / scaling

        PC = PC / scaling

    if geometry:
        data_train = np.concatenate((data_train, train_geom_data), axis=1)
        data_devel = np.concatenate((data_devel, devel_geom_data), axis=1)

    return data_train, labels_train, data_devel, labels_devel, raw_devel, PC, means, scaling


# Preparing the DISFA data
def Prepare_HOG_AU_data_generic_DISFA(train_recs, devel_recs, au, DISFA_dir, hog_data_dir, pca_loc, scale=False, geometry=False):

    # First extracting the labels
    au_train_dirs = [DISFA_dir + '/ActionUnit_Labels/' + user + '/' + user for user in train_recs]
    [labels_train, vid_ids_train] = extract_DISFA_labels(au_train_dirs, au)

    # Reading in the HOG data (of only relevant frames)
    [train_appearance_data, valid_ids_train_hog, vid_ids_train_string] =\
        Read_HOG_files_DISFA(train_recs, hog_data_dir)

    train_geom_data = Read_geom_files_DISFA(train_recs, hog_data_dir)

    # need to subsample every 3rd frame as now way too big

    if len(au) == 1:
        labels_train = labels_train[:, 0]

    reduced_inds = np.ones((labels_train.shape[0], ), dtype='bool')

    to_rem = np.round(np.linspace(0, reduced_inds.shape[0]-1, reduced_inds.shape[0]/1.5).astype('int32'))
    reduced_inds[to_rem] = False

    # also remove invalid ids based on CLM failing or AU not being labelled
    reduced_inds[valid_ids_train_hog == False] = False

    if len(au) == 1:
        labels_train = labels_train[reduced_inds]
    else:
        labels_train = labels_train[reduced_inds, :]

    train_appearance_data = train_appearance_data[reduced_inds, :]
    train_geom_data = train_geom_data[reduced_inds, :]

    # Extract devel data

    # First extracting the labels
    au_devel_dirs = [DISFA_dir + '/ActionUnit_Labels/' + user + '/' + user for user in devel_recs]
    [labels_devel, vid_ids_devel] = extract_DISFA_labels(au_devel_dirs, au)

    # Reading in the HOG data (of only relevant frames)
    [devel_appearance_data, valid_ids_devel_hog, vid_ids_devel_string] = \
         Read_HOG_files_DISFA(devel_recs, hog_data_dir)

    devel_geom_data = Read_geom_files_DISFA(devel_recs, hog_data_dir)

    devel_appearance_data = devel_appearance_data[1::3, :]
    labels_devel = labels_devel[1::3, :]
    devel_geom_data = devel_geom_data[1::3, :]

    # normalise the data
    dim_reds = scipy.io.loadmat(pca_loc)
    PC = dim_reds['PC']
    means = dim_reds['means_norm']
    scaling = dim_reds['stds_norm']

    # Grab all data for validation as want good params for all the data
    raw_devel = devel_appearance_data
    devel_appearance_data = (devel_appearance_data - means)
    devel_appearance_data = devel_appearance_data/scaling

    train_appearance_data = (train_appearance_data - means)/scaling

    data_train = np.dot(train_appearance_data, PC)
    data_devel = np.dot(devel_appearance_data, PC)

    if scale:

        # Some extra scaling
        scaling = np.std(data_train, axis=0)

        data_train = data_train / scaling
        data_devel = data_devel / scaling

        PC = PC / scaling

    if geometry:
        data_train = np.concatenate((data_train, train_geom_data), axis=1)
        data_devel = np.concatenate((data_devel, devel_geom_data), axis=1)

    return data_train, labels_train, data_devel, labels_devel, raw_devel, PC, means, scaling

# Preparing the BP4D data
def Prepare_HOG_AU_data_generic_DISFA_dynamic(train_recs, devel_recs, au, DISFA_dir, hog_data_dir, pca_loc, scale=False, geometry=False):

    # First extracting the labels
    au_train_dirs = [DISFA_dir + '/ActionUnit_Labels/' + user + '/' + user for user in train_recs]
    [labels_train, vid_ids_train] = extract_DISFA_labels(au_train_dirs, au)

    # Reading in the HOG data (of only relevant frames)
    [train_appearance_data, valid_ids_train_hog, vid_ids_train_string] =\
        Read_HOG_files_DISFA_dynamic(train_recs, hog_data_dir)

    train_geom_data = Read_geom_files_DISFA(train_recs, hog_data_dir)

    # need to subsample every 3rd frame as now way too big

    if len(au) == 1:
        labels_train = labels_train[:, 0]

    reduced_inds = np.ones((labels_train.shape[0], ), dtype='bool')

    to_rem = np.round(np.linspace(0, reduced_inds.shape[0]-1, reduced_inds.shape[0]/1.5).astype('int32'))
    reduced_inds[to_rem] = False

    # also remove invalid ids based on CLM failing or AU not being labelled
    reduced_inds[valid_ids_train_hog == False] = False

    if len(au) == 1:
        labels_train = labels_train[reduced_inds]
    else:
        labels_train = labels_train[reduced_inds, :]

    train_appearance_data = train_appearance_data[reduced_inds, :]
    train_geom_data = train_geom_data[reduced_inds, :]

    # Extract devel data

    # First extracting the labels
    au_devel_dirs = [DISFA_dir + '/ActionUnit_Labels/' + user + '/' + user for user in devel_recs]
    [labels_devel, vid_ids_devel] = extract_DISFA_labels(au_devel_dirs, au)

    # Reading in the HOG data (of only relevant frames)
    [devel_appearance_data, valid_ids_devel_hog, vid_ids_devel_string] = \
         Read_HOG_files_DISFA_dynamic(devel_recs, hog_data_dir)

    devel_geom_data = Read_geom_files_DISFA(devel_recs, hog_data_dir)

    devel_appearance_data = devel_appearance_data[1::3, :]
    labels_devel = labels_devel[1::3, :]
    devel_geom_data = devel_geom_data[1::3, :]

    # normalise the data
    dim_reds = scipy.io.loadmat(pca_loc)
    PC = dim_reds['PC']
    means = dim_reds['means_norm']
    scaling = dim_reds['stds_norm']

    # Grab all data for validation as want good params for all the data
    raw_devel = devel_appearance_data
    devel_appearance_data = (devel_appearance_data - means)
    devel_appearance_data = devel_appearance_data/scaling

    train_appearance_data = (train_appearance_data - means)/scaling

    data_train = np.dot(train_appearance_data, PC)
    data_devel = np.dot(devel_appearance_data, PC)

    if scale:

        # Some extra scaling
        scaling = np.std(data_train, axis=0)

        data_train = data_train / scaling
        data_devel = data_devel / scaling

        PC = PC / scaling

    if geometry:
        data_train = np.concatenate((data_train, train_geom_data), axis=1)
        data_devel = np.concatenate((data_devel, devel_geom_data), axis=1)

    return data_train, labels_train, data_devel, labels_devel, raw_devel, PC, means, scaling

# Preparing the BP4D data
def Prepare_HOG_AU_data_generic_BP4D(train_recs, devel_recs, au, BP4D_dir, hog_data_dir, pca_loc, geometry=False, scale=False):

    # First extracting the labels
    [labels_train, valid_ids_train, vid_ids_train] = extract_BP4D_labels(BP4D_dir, train_recs, au)

    if geometry:
        train_data_geom, files = Read_geom_files_BP4D(train_recs, hog_data_dir + '/train/')

    # Reading in the HOG data (of only relevant frames)
    [train_appearance_data, valid_ids_train_hog, vid_ids_train_string] = Read_HOG_files_BP4D(train_recs, hog_data_dir + '/train/')

    # Subsample the data to make training quicker
    labels_train = np.concatenate(labels_train)
    valid_ids_train = np.concatenate(valid_ids_train).astype('bool')

    if len(au) == 1:
        labels_train = labels_train[:, 0]

    reduced_inds = np.ones((labels_train.shape[0], ), dtype='bool')

    # only remove the data if single au used
    if len(au) == 1:
        # Remove two thirds of negative examples (to balance the training data a bit)
        inds_train = np.array(range(labels_train.shape[0]))
        neg_samples = inds_train[labels_train == 0]

        to_rem = neg_samples[np.round(np.linspace(0, neg_samples.shape[0]-1, neg_samples.shape[0]/1.5).astype('int32'))]
        reduced_inds[to_rem] = False
    
    # also remove invalid ids based on CLM failing or AU not being labelled
    reduced_inds[valid_ids_train == False] = False
    reduced_inds[valid_ids_train_hog == False] = False

    if len(au) == 1:
        labels_train = labels_train[reduced_inds]
    else:
        labels_train = labels_train[reduced_inds, :]

    train_appearance_data = train_appearance_data[reduced_inds, :]

    if geometry:
        train_data_geom = train_data_geom[reduced_inds, :]

    # Extract devel data
    
    # First extracting the labels
    [labels_devel, valid_ids_devel, vid_ids_devel] = extract_BP4D_labels(BP4D_dir, devel_recs, au)
    
    # Reading in the HOG data (of only relevant frames)    
    [devel_appearance_data, valid_ids_devel_hog, vid_ids_devel_string] = \
        Read_HOG_files_BP4D(devel_recs, hog_data_dir + '/devel/')

    if geometry:
        devel_data_geom, files = Read_geom_files_BP4D(devel_recs, hog_data_dir + '/devel/')

    labels_devel = np.concatenate(labels_devel)
    
    # normalise the data
    dim_reds = scipy.io.loadmat(pca_loc)
    PC = dim_reds['PC']
    means = dim_reds['means_norm']
    scaling = dim_reds['stds_norm']
     
    # Grab all data for validation as want good params for all the data
    raw_devel = devel_appearance_data
    devel_appearance_data = (devel_appearance_data - means)
    devel_appearance_data = devel_appearance_data/scaling

    train_appearance_data = (train_appearance_data - means)/scaling
        
    data_train = np.dot(train_appearance_data, PC)
    data_devel = np.dot(devel_appearance_data, PC)

    if scale:

        # Some extra scaling
        scaling = np.std(data_train, axis=0)

        data_train = data_train / scaling
        data_devel = data_devel / scaling

        PC = PC / scaling

    if geometry:
        data_train = np.concatenate((data_train, train_data_geom), axis=1)
        data_devel = np.concatenate((data_devel, devel_data_geom), axis=1)

    return data_train, labels_train, data_devel, labels_devel, raw_devel, PC, means, scaling

# Preparing the BP4D data
def Prepare_HOG_AU_data_generic_BP4D_intensity(train_recs, devel_recs, au, BP4D_dir, hog_data_dir, pca_loc, geometry=False, scale=False):

    # First extracting the labels
    [labels_train, valid_ids_train, vid_ids_train, _] = extract_BP4D_labels_intensity(BP4D_dir, train_recs, au)

    if geometry:
        train_data_geom, files = Read_geom_files_BP4D(train_recs, hog_data_dir + '/train/')

    # Reading in the HOG data (of only relevant frames)
    [train_appearance_data, valid_ids_train_hog, vid_ids_train_string] = Read_HOG_files_BP4D(train_recs, hog_data_dir + '/train/')

    # Subsample the data to make training quicker
    labels_train = np.concatenate(labels_train)
    valid_ids_train = np.concatenate(valid_ids_train).astype('bool')

    valid_ids_train = valid_ids_train[:,0]

    if len(au) == 1:
        labels_train = labels_train[:, 0]

    reduced_inds = np.ones((labels_train.shape[0], ), dtype='bool')

    # also remove invalid ids based on CLM failing or AU not being labelled
    reduced_inds[valid_ids_train == False] = False
    reduced_inds[valid_ids_train_hog == False] = False

    if len(au) == 1:
        labels_train = labels_train[reduced_inds]
    else:
        labels_train = labels_train[reduced_inds, :]

    train_appearance_data = train_appearance_data[reduced_inds, :]

    if geometry:
        train_data_geom = train_data_geom[reduced_inds, :]

    # Extract devel data

    # First extracting the labels
    [labels_devel, valid_ids_devel, vid_ids_devel, _] = extract_BP4D_labels_intensity(BP4D_dir, devel_recs, au)

    # Reading in the HOG data (of only relevant frames)
    [devel_appearance_data, valid_ids_devel_hog, vid_ids_devel_string] = \
        Read_HOG_files_BP4D(devel_recs, hog_data_dir + '/devel/')

    if geometry:
        devel_data_geom, files = Read_geom_files_BP4D(devel_recs, hog_data_dir + '/devel/')

    labels_devel = np.concatenate(labels_devel)

    reduced_inds = np.ones((labels_devel.shape[0], ), dtype='bool')

    valid_ids_devel = np.concatenate(valid_ids_devel).astype('bool')
    valid_ids_devel = valid_ids_devel[:,0]

    # also remove invalid ids based on CLM failing or AU not being labelled
    reduced_inds[valid_ids_devel == False] = False

    if len(au) == 1:
        labels_devel = labels_devel[reduced_inds]
    else:
        labels_devel = labels_devel[reduced_inds, :]

    devel_appearance_data = devel_appearance_data[reduced_inds, :]
    devel_data_geom = devel_data_geom[reduced_inds, :]

    # normalise the data
    dim_reds = scipy.io.loadmat(pca_loc)
    PC = dim_reds['PC']
    means = dim_reds['means_norm']
    scaling = dim_reds['stds_norm']

    # Grab all data for validation as want good params for all the data
    raw_devel = devel_appearance_data
    devel_appearance_data = (devel_appearance_data - means)
    devel_appearance_data = devel_appearance_data/scaling

    train_appearance_data = (train_appearance_data - means)/scaling

    data_train = np.dot(train_appearance_data, PC)
    data_devel = np.dot(devel_appearance_data, PC)

    if scale:

        # Some extra scaling
        scaling = np.std(data_train, axis=0)

        data_train = data_train / scaling
        data_devel = data_devel / scaling

        PC = PC / scaling

    if geometry:
        data_train = np.concatenate((data_train, train_data_geom), axis=1)
        data_devel = np.concatenate((data_devel, devel_data_geom), axis=1)

    return data_train, labels_train, data_devel, labels_devel, raw_devel, PC, means, scaling


# Preparing the BP4D data
def Prepare_HOG_AU_data_generic_BP4D_no_PCA(train_recs, devel_recs, au, BP4D_dir, hog_data_dir):

    # First extracting the labels
    [labels_train, valid_ids_train, vid_ids_train] = extract_BP4D_labels(BP4D_dir, train_recs, au)

    # Reading in the HOG data (of only relevant frames)
    [train_appearance_data, valid_ids_train_hog, vid_ids_train_string] = Read_HOG_files_BP4D(train_recs, hog_data_dir + '/train/')

    # Subsample the data to make training quicker
    labels_train = np.concatenate(labels_train)
    valid_ids_train = np.concatenate(valid_ids_train).astype('bool')

    if len(au) == 1:
        labels_train = labels_train[:, 0]

    reduced_inds = np.ones((labels_train.shape[0], ), dtype='bool')

    # only remove the data if single au used
    if len(au) == 1:
        # Remove two thirds of negative examples (to balance the training data a bit)
        inds_train = np.array(range(labels_train.shape[0]))
        neg_samples = inds_train[labels_train == 0]

        to_rem = neg_samples[np.round(np.linspace(0, neg_samples.shape[0]-1, neg_samples.shape[0]/1.5).astype('int32'))]
        reduced_inds[to_rem] = False

    # also remove invalid ids based on CLM failing or AU not being labelled
    reduced_inds[valid_ids_train == False] = False
    reduced_inds[valid_ids_train_hog == False] = False

    if len(au) == 1:
        labels_train = labels_train[reduced_inds]
    else:
        labels_train = labels_train[reduced_inds, :]

    train_appearance_data = train_appearance_data[reduced_inds, :]

    # Extract devel data

    # First extracting the labels
    [labels_devel, valid_ids_devel, vid_ids_devel] = extract_BP4D_labels(BP4D_dir, devel_recs, au)

    # Reading in the HOG data (of only relevant frames)
    [devel_appearance_data, valid_ids_devel_hog, vid_ids_devel_string] = \
        Read_HOG_files_BP4D(devel_recs, hog_data_dir + '/devel/')

    labels_devel = np.concatenate(labels_devel)

    data_train = train_appearance_data
    data_devel = devel_appearance_data

    return data_train, labels_train, data_devel, labels_devel


def Read_HOG_files_BP4D_dynamic(users, hog_data_dir):
    
    import glob
    import struct
    import numpy as np
    
    vid_id = []
    valid_inds = []
    
    feats_filled = 0
    hog_data = np.array((0,0))
    
    for i in range(len(users)):
        
        hog_files = glob.glob(hog_data_dir + users[i] + '*.hog')
        
        start_person_ind = feats_filled
        
        for h in range(len(hog_files)):

            hog_file = hog_files[h]
            
            f = open(hog_file, 'rb')

            curr_data = []
            curr_ind = 0
            try:
                while True:

                    if curr_ind == 0:
                        
                        a = f.read(4)

                        if not a:
                            break

                        num_cols = struct.unpack('i', a)[0]
    
                        num_rows = struct.unpack('i', f.read(4))[0]
                        num_chan = struct.unpack('i', f.read(4))[0]
    
                        # preallocate some space
                        if curr_ind == 0:
                            curr_data = np.zeros((1000, 1 + num_rows * num_cols * num_chan))
                            num_feats = 1 + num_rows * num_cols * num_chan
                        
                        # Add more spce to the buffer
                        if curr_ind >= curr_data.shape[0]:
                            curr_data = np.concatenate(curr_data, np.zeros((1000, 1 + num_rows * num_cols * num_chan)))

                        feature_vec = np.fromfile(f, dtype='float32', count = 1 + num_rows * num_cols * num_chan)
                        curr_data[curr_ind, :] = feature_vec

                        curr_ind += 1
                            
                    else:
    
                        # Reading in batches of 5000
                        feature_vec = np.fromfile(f, dtype='float32', count=(3 + num_feats) * 5000)
                        if feature_vec.shape[0] == 0:
                            break

                        feature_vec.shape = (feature_vec.shape[0]/(3+num_feats), 3 + num_feats)
                                                        
                        feature_vec = feature_vec[:, 3:]

                        num_rows_read = feature_vec.shape[0]
    
                        curr_data[curr_ind:curr_ind+num_rows_read, :] = feature_vec
    
                        curr_ind = curr_ind + feature_vec.shape[0]
            finally:
                f.close()
                
            curr_data = curr_data[0:curr_ind, :]
            vid_id_curr = [users[i]] * curr_ind

            vid_id.append(vid_id_curr)

            # Assume same number of frames per video
            if i == 0 and h == 0:
                hog_data = np.zeros((curr_ind * len(users) * 8, num_feats))

            if hog_data.shape[0] < feats_filled+curr_ind:
                hog_data = np.concatenate(hog_data, np.zeros(hog_data.shape[0], num_feats))

            hog_data[feats_filled:feats_filled+curr_ind, :] = curr_data

            feats_filled = feats_filled + curr_ind

        # Subtract the median for a dynamic model        
        person_ids = range(start_person_ind, feats_filled)
        hog_data[person_ids, 1:] = hog_data[person_ids, 1:] - np.median(hog_data[person_ids, 1:], axis=0)
    
    if hog_data.shape[0] != 0:
        valid_inds = hog_data[0:feats_filled, 0]
        hog_data = hog_data[0:feats_filled, 1:]

    return hog_data, valid_inds, vid_id

# Preparing the BP4D data
def Prepare_HOG_AU_data_generic_BP4D_dynamic(train_recs, devel_recs, au, BP4D_dir, hog_data_dir, pca_loc, scale=False, geometry=True):
    
    import numpy as np
    import scipy.io
    
    # First extracting the labels
    [ labels_train, valid_ids_train, vid_ids_train ] = extract_BP4D_labels(BP4D_dir, train_recs, au)
    
    # Reading in the HOG data (of only relevant frames)
    [train_appearance_data, valid_ids_train_hog, vid_ids_train_string] = Read_HOG_files_BP4D_dynamic(train_recs, hog_data_dir + '/train/')

    if geometry:
        train_data_geom, files = Read_geom_files_BP4D(train_recs, hog_data_dir + '/train/')

    # Subsample the data to make training quicker
    labels_train = np.concatenate(labels_train)
    valid_ids_train = np.concatenate(valid_ids_train).astype('bool')

    if len(au) == 1:
        labels_train = labels_train[:, 0]

    reduced_inds = np.ones((labels_train.shape[0], ), dtype='bool')

    # only remove the data if single au used
    if len(au) == 1:
        # Remove two thirds of negative examples (to balance the training data a bit)
        inds_train = np.array(range(labels_train.shape[0]))
        neg_samples = inds_train[labels_train == 0]

        to_rem = neg_samples[np.round(np.linspace(0,neg_samples.shape[0]-1, neg_samples.shape[0]/1.5).astype('int32'))]
        reduced_inds[to_rem] = False

    # also remove invalid ids based on CLM failing or AU not being labelled
    reduced_inds[valid_ids_train == False] = False
    reduced_inds[valid_ids_train_hog == False] = False

    if len(au) == 1:
        labels_train = labels_train[reduced_inds]
    else:
        labels_train = labels_train[reduced_inds, :]

    train_appearance_data = train_appearance_data[reduced_inds, :]

    if geometry:
        train_data_geom = train_data_geom[reduced_inds, :]

    # Extract devel data
    
    # First extracting the labels
    [labels_devel, valid_ids_devel, vid_ids_devel] = extract_BP4D_labels(BP4D_dir, devel_recs, au)
    
    # Reading in the HOG data (of only relevant frames)    
    [devel_appearance_data, valid_ids_devel_hog, vid_ids_devel_string] =\
        Read_HOG_files_BP4D_dynamic(devel_recs, hog_data_dir + '/devel/')
    
    labels_devel = np.concatenate(labels_devel)

    if geometry:
        devel_data_geom, files = Read_geom_files_BP4D(devel_recs, hog_data_dir + '/devel/')


    # normalise the data
    dim_reds = scipy.io.loadmat(pca_loc)
    PC = dim_reds['PC']
    means = dim_reds['means_norm']
    scaling = dim_reds['stds_norm']
     
    # Grab all data for validation as want good params for all the data
    raw_devel = devel_appearance_data
    devel_appearance_data = (devel_appearance_data - means)
    devel_appearance_data = devel_appearance_data/scaling

    train_appearance_data = (train_appearance_data - means)/scaling
        
    data_train = np.dot(train_appearance_data, PC)
    data_devel = np.dot(devel_appearance_data, PC)

    if scale:

        # Some extra scaling
        scaling = np.std(data_train, axis=0)

        data_train = data_train / scaling
        data_devel = data_devel / scaling

        PC = PC / scaling

    if geometry:
        data_train = np.concatenate((data_train, train_data_geom), axis=1)
        data_devel = np.concatenate((data_devel, devel_data_geom), axis=1)

    return data_train, labels_train, data_devel, labels_devel, raw_devel, PC, means, scaling
