# The SVM baseline for BO4D
import shared_defs_BP4D
import data_preparation

import logistic_regression

(all_aus, train_recs, devel_recs, BP4D_dir, hog_data_dir) = shared_defs_BP4D.shared_defs();

pca_loc = "../pca_generation/generic_face_rigid";

f = open("./trained/BP4D_train_static_log_reg.txt", 'w');

for au in all_aus:

    #hyperparams = {"C":[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], "validate_params":["C"]}
    #hyperparams = {"C":[1], "validate_params":["C"]}

    # load the training and testing data for the current fold
    [train_samples, train_labels, valid_samples, valid_labels, raw_valid, PC, means, scaling] = data_preparation.Prepare_HOG_AU_data_generic_BP4D(train_recs, devel_recs, [au], BP4D_dir, hog_data_dir, pca_loc);

    import validation_helpers

    train_fn = logistic_regression.train_log_reg
    test_fn = logistic_regression.test_log_reg

    hyperparams = {'batch_size':100, 'learning_rate':0.1, 'n_epochs':10}

    # Cross-validate here
    best_params, all_params = validation_helpers.validate_grid_search(train_fn, test_fn, False, train_samples, train_labels, valid_samples, valid_labels, hyperparams);

    model = train_fn(train_labels, train_samples, best_params);

    f1, precision, recall, prediction = test_fn(valid_labels, valid_samples, model);

    print 'AU%d done: prec %.4f, recall %.4f, f1 %.4f\n' % (au, precision, recall, f1)

    f.write("%d %.4f %.4f %.4f\n" % (au, precision, recall, f1))

f.close()
