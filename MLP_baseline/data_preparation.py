import numpy as np
import scipy.io

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

            if labels_rec_all == None:

                labels_rec_all = labels_rec
                valid_ids_rec_all = valid_ids_rec

            else:

                labels_rec_all = np.concatenate((labels_rec_all, labels_rec), axis=1)
                valid_ids_rec_all = np.logical_and(valid_ids_rec_all, vid_ids_rec)

            vid_ids[ind, :] = [vid_ids_rec[0], vid_ids_rec[1]]

        labels.append(labels_rec_all)
        valid_ids.append(valid_ids_rec_all)
        ind = ind + 1

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

def Read_HOG_files_BP4D(users, hog_data_dir):
    
    import glob
    import struct
    import numpy as np
    
    vid_id = []
    valid_inds = []
    
    feats_filled = 0
    hog_data = np.array((0,0))
    
    for i in range(len(users)):
        
        hog_files = glob.glob(hog_data_dir + users[i] + '*.hog')
        
        for h in range(len(hog_files)):

            hog_file = hog_files[h]
            
            f = open(hog_file, 'rb')

            curr_data = []
            curr_ind = 0
            try:
                while(True):

                    if(curr_ind == 0):
                        
                        a =  f.read(4)

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
                        if(curr_ind >= curr_data.shape[0]):
                            curr_data = np.concatenate(curr_data, np.zeros((1000, 1 + num_rows * num_cols * num_chan)))

                        feature_vec = np.fromfile(f, dtype='float32', count = 1 + num_rows * num_cols * num_chan)
                        curr_data[curr_ind, :] = feature_vec;

                        curr_ind = curr_ind + 1
                            
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
                f.close();
                
            curr_data = curr_data[0:curr_ind,:]
            vid_id_curr = [users[i]] * curr_ind

            vid_id.append(vid_id_curr)


            # Assume same number of frames per video
            if(i==0 and h == 0):
                hog_data = np.zeros((curr_ind * len(users) * 8, num_feats))

            if(hog_data.shape[0] < feats_filled+curr_ind):
               hog_data = np.concatenate(hog_data, np.zeros(hog_data.shape[0], num_feats))

            hog_data[feats_filled:feats_filled+curr_ind,:] = curr_data

            feats_filled = feats_filled + curr_ind
    
    if hog_data.shape[0] != 0:
        valid_inds = hog_data[0:feats_filled, 0]
        hog_data = hog_data[0:feats_filled, 1:]

    return (hog_data, valid_inds, vid_id)

def Read_HOG_files_SEMAINE(users, vid_ids, hog_data_dir):

    import struct
    import numpy as np

    vid_id = []
    valid_inds = []

    feats_filled = 0
    hog_data = np.array((0,0))

    for i in range(len(users)):

        hog_file = hog_data_dir + '/' + users[i] + '.hog'

        f = open(hog_file, 'rb')

        a =  f.read(4)

        if(not a):
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

# Preparing the SEMAINE data
def Prepare_HOG_AU_data_generic_SEMAINE(train_recs, devel_recs, au, SEMAINE_dir, hog_data_dir, pca_loc):

    # First extracting the labels
    SEMAINE_label_dir = '../SEMAINE_baseline/training_labels/'

    [labels_train, valid_ids_train, vid_ids_train] = extract_SEMAINE_labels(SEMAINE_label_dir, train_recs, au)

    # Reading in the HOG data (of only relevant frames)
    [train_appearance_data, valid_ids_train_hog, vid_ids_train_string] = Read_HOG_files_SEMAINE(train_recs, vid_ids_train, hog_data_dir + '/train/')

    # Subsample the data to make training quicker
    labels_train = np.concatenate(labels_train)
    valid_ids_train = np.concatenate(valid_ids_train).astype('bool')

    if len(au) == 1:
        labels_train = labels_train[:, 0]

    valid_ids_train = valid_ids_train[:,0]

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
    [labels_devel, valid_ids_devel, vid_ids_devel] = extract_SEMAINE_labels(SEMAINE_label_dir, devel_recs, au)

    # Reading in the HOG data (of only relevant frames)
    [devel_appearance_data, valid_ids_devel_hog, vid_ids_devel_string] = \
        Read_HOG_files_SEMAINE(devel_recs, vid_ids_devel, hog_data_dir + '/devel/')

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

    return data_train, labels_train, data_devel, labels_devel, raw_devel, PC, means, scaling


# Preparing the BP4D data
def Prepare_HOG_AU_data_generic_BP4D(train_recs, devel_recs, au, BP4D_dir, hog_data_dir, pca_loc):

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
def Prepare_HOG_AU_data_generic_BP4D_dynamic(train_recs, devel_recs, au, BP4D_dir, hog_data_dir, pca_loc):
    
    import numpy as np
    import scipy.io
    
    # First extracting the labels
    [ labels_train, valid_ids_train, vid_ids_train ] = extract_BP4D_labels(BP4D_dir, train_recs, au)
    
    # Reading in the HOG data (of only relevant frames)
    [train_appearance_data, valid_ids_train_hog, vid_ids_train_string] = Read_HOG_files_BP4D_dynamic(train_recs, hog_data_dir + '/train/')

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
    # Extract devel data
    
    # First extracting the labels
    [labels_devel, valid_ids_devel, vid_ids_devel] = extract_BP4D_labels(BP4D_dir, devel_recs, au)
    
    # Reading in the HOG data (of only relevant frames)    
    [devel_appearance_data, valid_ids_devel_hog, vid_ids_devel_string] =\
        Read_HOG_files_BP4D_dynamic(devel_recs, hog_data_dir + '/devel/')
    
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

    return data_train, labels_train, data_devel, labels_devel, raw_devel, PC, means, scaling
