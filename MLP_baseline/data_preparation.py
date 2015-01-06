def extract_BP4D_labels(BP4D_dir, recs, aus):
    
    import glob
    import os
    import numpy as np
    
    aus_BP4D_all = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23];
    
    inds_to_use = [];
    
    for au in aus:    
        inds_to_use = inds_to_use + [aus_BP4D_all.index(au)];
        
    num_files = len(glob.glob(BP4D_dir + '/*.csv'));
    
    labels = [];
    valid_ids = [];
    # This should be ndarray
    vid_ids = np.zeros((num_files, 2));
    filenames = [];
    
    file_id = 0;
    
    for i in range(len(recs)):

        csvs = glob.glob(BP4D_dir + '/' + recs[i] + '*.csv');
    
        for f in range(len(csvs)):
            
            filepath = csvs[f];

            _, filename_full = os.path.split(csvs[f])
            filename,_ = os.path.splitext(filename_full)
            
            filenames = filenames + [filename];
            OCC = np.genfromtxt(filepath, delimiter=',')

            frame_nums = OCC[1:,0]; #get all frame numbers
            codes = OCC[1:,1:]; #get codes for all action units
            occlusions = OCC[1:,-1];
            
            codes = codes[:, np.array(aus_BP4D_all) - 1];
            
            # Finding the invalid regions
            valid = occlusions != 1;                     
            
            for s in range(codes.shape[1]):              
                valid = np.logical_and(valid, (codes[:,s] != 9));
                            
            vid_ids[file_id,:] = [frame_nums[0], frame_nums[-1]];

            labels.append(codes[:, inds_to_use]);
            valid_ids.append(valid);
            
            file_id = file_id + 1;
            
    vid_ids = vid_ids[0:file_id,:];
    return (labels, valid_ids, vid_ids)

def Read_HOG_files_BP4D(users, hog_data_dir):
    
    import glob
    import struct
    import numpy as np
    
    vid_id = [];
    valid_inds = [];
    
    feats_filled = 0;
    hog_data = np.array((0,0));
    
    for i in range(len(users)):
        
        hog_files = glob.glob(hog_data_dir + users[i] + '*.hog');
        
        for h in range(len(hog_files)):

            hog_file = hog_files[h];
            
            f = open(hog_file, 'rb');

            curr_data = [];
            curr_ind = 0;
            try:
                while(True):

                    if(curr_ind == 0):
                        
                        a =  f.read(4);

                        if(not a):
                            break;

                        num_cols = struct.unpack('i', a)[0];
    
                        num_rows = struct.unpack('i', f.read(4))[0];
                        num_chan = struct.unpack('i', f.read(4))[0];
    
                        # preallocate some space
                        if(curr_ind == 0):
                            curr_data = np.zeros((1000, 1 + num_rows * num_cols * num_chan));
                            num_feats =  1 + num_rows * num_cols * num_chan;
                        
                        # Add more spce to the buffer
                        if(curr_ind >= curr_data.shape[0]):
                            curr_data = np.concatenate(curr_data, np.zeros((1000, 1 + num_rows * num_cols * num_chan)));

                        feature_vec = np.fromfile(f, dtype='float32', count = 1 + num_rows * num_cols * num_chan);
                        curr_data[curr_ind, :] = feature_vec;

                        curr_ind = curr_ind + 1;            
                            
                    else:
    
                        # Reading in batches of 5000
    
                        feature_vec = np.fromfile(f, dtype='float32', count=(3 + num_feats) * 5000);
                        if(feature_vec.shape[0]==0):
                            break;

                        feature_vec.shape = (feature_vec.shape[0]/(3+num_feats), 3 + num_feats)
                                                        
                        feature_vec = feature_vec[:,3:];

                        num_rows_read = feature_vec.shape[0];
    
                        curr_data[curr_ind:curr_ind+num_rows_read,:] = feature_vec;
    
                        curr_ind = curr_ind + feature_vec.shape[0];
            finally:
                f.close();
                
            curr_data = curr_data[0:curr_ind,:];
            vid_id_curr = [users[i]] * curr_ind;

            vid_id.append(vid_id_curr);


            # Assume same number of frames per video
            if(i==0 and h == 0):
                hog_data = np.zeros((curr_ind * len(users) * 8, num_feats));

            if(hog_data.shape[0] < feats_filled+curr_ind):
               hog_data = np.concatenate(hog_data, np.zeros(hog_data.shape[0], num_feats));

            hog_data[feats_filled:feats_filled+curr_ind,:] = curr_data;

            feats_filled = feats_filled + curr_ind;
    
    if(hog_data.shape[0] != 0):        
        valid_inds = hog_data[0:feats_filled,0];
        hog_data = hog_data[0:feats_filled,1:];     

    return (hog_data, valid_inds, vid_id)

# Preparing the BP4D data
def Prepare_HOG_AU_data_generic_BP4D(train_recs, devel_recs, au, BP4D_dir, hog_data_dir, pca_loc):
    
    import numpy as np;
    import scipy.io
    
    # First extracting the labels
    [ labels_train, valid_ids_train, vid_ids_train ] = extract_BP4D_labels(BP4D_dir, train_recs, au);
    
    # Reading in the HOG data (of only relevant frames)
    [train_appearance_data, valid_ids_train_hog, vid_ids_train_string] = Read_HOG_files_BP4D(train_recs, hog_data_dir + '/train/');
    
    # Subsample the data to make training quicker
    labels_train = np.concatenate(labels_train);
    valid_ids_train = np.concatenate(valid_ids_train).astype('bool');
    
    # Remove two thirds of negative examples (to balance the training data a bit)
    inds_train = np.array(range(labels_train.shape[0]));

    labels_train = labels_train[:,0];
    neg_samples = inds_train[labels_train == 0];
    
    reduced_inds = np.ones(labels_train.shape, dtype='bool');
    
    to_rem = neg_samples[np.round(np.linspace(0,neg_samples.shape[0]-1, neg_samples.shape[0]/1.5).astype('int32'))];
    reduced_inds[to_rem] = False;
    
    # also remove invalid ids based on CLM failing or AU not being labelled
    reduced_inds[valid_ids_train == False] = False;
    reduced_inds[valid_ids_train_hog == False] = False;
    
    labels_train = labels_train[reduced_inds];
    train_appearance_data = train_appearance_data[reduced_inds,:];
    
    #print vid_ids_train_string.shape
    #vid_ids_train_string = vid_ids_train_string[reduced_inds,:];
    
    # Extract devel data
    
    # First extracting the labels
    [ labels_devel, valid_ids_devel, vid_ids_devel ] = extract_BP4D_labels(BP4D_dir, devel_recs, au);
    
    # Reading in the HOG data (of only relevant frames)    
    [devel_appearance_data, valid_ids_devel_hog, vid_ids_devel_string] = Read_HOG_files_BP4D(devel_recs, hog_data_dir + '/devel/');
    
    labels_devel = np.concatenate(labels_devel);
    
    # normalise the data
    dim_reds = scipy.io.loadmat(pca_loc);
    PC = dim_reds['PC'];
    means = dim_reds['means_norm'];
    scaling = dim_reds['stds_norm'];
     
    # Grab all data for validation as want good params for all the data
    raw_devel = devel_appearance_data;
    devel_appearance_data = (devel_appearance_data - means);
    devel_appearance_data = devel_appearance_data/scaling;

    train_appearance_data = (train_appearance_data - means)/scaling;
        
    data_train = np.dot(train_appearance_data, PC);
    data_devel = np.dot(devel_appearance_data, PC);

    return (data_train, labels_train, data_devel, labels_devel, raw_devel, PC, means, scaling)

def Read_HOG_files_BP4D_dynamic(users, hog_data_dir):
    
    import glob
    import struct
    import numpy as np
    
    vid_id = [];
    valid_inds = [];
    
    feats_filled = 0;
    hog_data = np.array((0,0));
    
    for i in range(len(users)):
        
        hog_files = glob.glob(hog_data_dir + users[i] + '*.hog');
        
        start_person_ind = feats_filled;
        
        for h in range(len(hog_files)):

            hog_file = hog_files[h];
            
            f = open(hog_file, 'rb');

            curr_data = [];
            curr_ind = 0;
            try:
                while(True):

                    if(curr_ind == 0):
                        
                        a =  f.read(4);

                        if(not a):
                            break;

                        num_cols = struct.unpack('i', a)[0];
    
                        num_rows = struct.unpack('i', f.read(4))[0];
                        num_chan = struct.unpack('i', f.read(4))[0];
    
                        # preallocate some space
                        if(curr_ind == 0):
                            curr_data = np.zeros((1000, 1 + num_rows * num_cols * num_chan));
                            num_feats =  1 + num_rows * num_cols * num_chan;
                        
                        # Add more spce to the buffer
                        if(curr_ind >= curr_data.shape[0]):
                            curr_data = np.concatenate(curr_data, np.zeros((1000, 1 + num_rows * num_cols * num_chan)));

                        feature_vec = np.fromfile(f, dtype='float32', count = 1 + num_rows * num_cols * num_chan);
                        curr_data[curr_ind, :] = feature_vec;

                        curr_ind = curr_ind + 1;            
                            
                    else:
    
                        # Reading in batches of 5000
    
                        feature_vec = np.fromfile(f, dtype='float32', count=(3 + num_feats) * 5000);
                        if(feature_vec.shape[0]==0):
                            break;

                        feature_vec.shape = (feature_vec.shape[0]/(3+num_feats), 3 + num_feats)
                                                        
                        feature_vec = feature_vec[:,3:];

                        num_rows_read = feature_vec.shape[0];
    
                        curr_data[curr_ind:curr_ind+num_rows_read,:] = feature_vec;
    
                        curr_ind = curr_ind + feature_vec.shape[0];
            finally:
                f.close();
                
            curr_data = curr_data[0:curr_ind,:];
            vid_id_curr = [users[i]] * curr_ind;

            vid_id.append(vid_id_curr);


            # Assume same number of frames per video
            if(i==0 and h == 0):
                hog_data = np.zeros((curr_ind * len(users) * 8, num_feats));

            if(hog_data.shape[0] < feats_filled+curr_ind):
               hog_data = np.concatenate(hog_data, np.zeros(hog_data.shape[0], num_feats));

            hog_data[feats_filled:feats_filled+curr_ind,:] = curr_data;

            feats_filled = feats_filled + curr_ind;

        # Subtract the median for a dynamic model        
        person_ids = range(start_person_ind, feats_filled);        
        hog_data[person_ids,1:] = hog_data[person_ids,1:] - np.median(hog_data[person_ids,1:], axis=0)
    
    if(hog_data.shape[0] != 0):        
        valid_inds = hog_data[0:feats_filled,0];
        hog_data = hog_data[0:feats_filled,1:];     

    return (hog_data, valid_inds, vid_id)

# Preparing the BP4D data
def Prepare_HOG_AU_data_generic_BP4D_dynamic(train_recs, devel_recs, au, BP4D_dir, hog_data_dir, pca_loc):
    
    import numpy as np;
    import scipy.io
    
    # First extracting the labels
    [ labels_train, valid_ids_train, vid_ids_train ] = extract_BP4D_labels(BP4D_dir, train_recs, au);
    
    # Reading in the HOG data (of only relevant frames)
    [train_appearance_data, valid_ids_train_hog, vid_ids_train_string] = Read_HOG_files_BP4D_dynamic(train_recs, hog_data_dir + '/train/');
    
    # Subsample the data to make training quicker
    labels_train = np.concatenate(labels_train);
    valid_ids_train = np.concatenate(valid_ids_train).astype('bool');
    
    # Remove two thirds of negative examples (to balance the training data a bit)
    inds_train = np.array(range(labels_train.shape[0]));

    labels_train = labels_train[:,0];
    neg_samples = inds_train[labels_train == 0];
    
    reduced_inds = np.ones(labels_train.shape, dtype='bool');
    
    to_rem = neg_samples[np.round(np.linspace(0,neg_samples.shape[0]-1, neg_samples.shape[0]/1.5).astype('int32'))];
    reduced_inds[to_rem] = False;
    
    # also remove invalid ids based on CLM failing or AU not being labelled
    reduced_inds[valid_ids_train == False] = False;
    reduced_inds[valid_ids_train_hog == False] = False;
    
    labels_train = labels_train[reduced_inds];
    train_appearance_data = train_appearance_data[reduced_inds,:];
    
    #print vid_ids_train_string.shape
    #vid_ids_train_string = vid_ids_train_string[reduced_inds,:];
    
    # Extract devel data
    
    # First extracting the labels
    [ labels_devel, valid_ids_devel, vid_ids_devel ] = extract_BP4D_labels(BP4D_dir, devel_recs, au);
    
    # Reading in the HOG data (of only relevant frames)    
    [devel_appearance_data, valid_ids_devel_hog, vid_ids_devel_string] = Read_HOG_files_BP4D_dynamic(devel_recs, hog_data_dir + '/devel/');
    
    labels_devel = np.concatenate(labels_devel);
    
    # normalise the data
    dim_reds = scipy.io.loadmat(pca_loc);
    PC = dim_reds['PC'];
    means = dim_reds['means_norm'];
    scaling = dim_reds['stds_norm'];
     
    # Grab all data for validation as want good params for all the data
    raw_devel = devel_appearance_data;
    devel_appearance_data = (devel_appearance_data - means);
    devel_appearance_data = devel_appearance_data/scaling;

    train_appearance_data = (train_appearance_data - means)/scaling;
        
    data_train = np.dot(train_appearance_data, PC);
    data_devel = np.dot(devel_appearance_data, PC);

    return (data_train, labels_train, data_devel, labels_devel, raw_devel, PC, means, scaling)


# Preparing the SEMAINE data
#def Prepare_HOG_AU_data_generic_SEMAINE(train_recs, devel_recs, au, SEMAINE_dir, hog_data_dir, pca_loc):
    
    