# The SVM baseline for BO4D
import shared_defs_DISFA
import data_preparation

(all_aus, users, DISFA_dir, hog_data_dir) = shared_defs_DISFA.shared_defs()

train_recs = users[0:int(len(users)/2)]
devel_recs = users[(int(len(users)/2)):]

pca_loc = "../pca_generation/generic_face_rigid"

f = open("./trained/DISFA_train_dynamic_lin_svm.txt", 'w')

for au in all_aus:
               
    hyperparams = {"C": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], "validate_params": ["C"]}
    
    # load the training and testing data for the current fold
    [train_samples, train_labels, valid_samples, valid_labels, raw_valid, PC, means, scaling] = \
        data_preparation.Prepare_HOG_AU_data_generic_DISFA_dynamic(train_recs, devel_recs, [au],
                                                             DISFA_dir, hog_data_dir, pca_loc)
    train_labels[train_labels > 1] = 1
    valid_labels[valid_labels > 1] = 1

    import linear_SVM
    import validation_helpers
    
    train_fn = linear_SVM.train_SVM
    test_fn = linear_SVM.test_SVM

    # Cross-validate here                
    best_params, all_params = validation_helpers.validate_grid_search(train_fn, test_fn, False, train_samples,
                                                                      train_labels, valid_samples, valid_labels,
                                                                      hyperparams)

    model = train_fn(train_labels, train_samples, best_params)

    f1, precision, recall, prediction = test_fn(valid_labels, valid_samples, model)

    print 'AU%d done: prec %.4f, recall %.4f, f1 %.4f\n' % (au, precision, recall, f1)

    f.write("%d %.4f %.4f %.4f\n" % (au, precision, recall, f1))
    
f.close()
