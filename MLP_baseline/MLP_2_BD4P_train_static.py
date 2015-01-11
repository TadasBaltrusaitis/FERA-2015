# The SVM baseline for BO4D
import shared_defs_BP4D
import data_preparation
import numpy

import mlp_multi_layer

(all_aus, train_recs, devel_recs, BP4D_dir, hog_data_dir) = shared_defs_BP4D.shared_defs()

pca_loc = "../pca_generation/generic_face_rigid"

f = open("./trained/BP4D_train_mlp_2.txt", 'w')

for au in all_aus:

    # load the training and testing data for the current fold
    [train_samples, train_labels, valid_samples, valid_labels, raw_valid, PC, means, scaling] = \
        data_preparation.Prepare_HOG_AU_data_generic_BP4D(train_recs, devel_recs, [au], BP4D_dir, hog_data_dir, pca_loc)

    import validation_helpers

    train_fn = mlp_multi_layer.train_mlp
    test_fn = mlp_multi_layer.test_mlp

    hyperparams = {
        'batch_size': [200],
        'learning_rate': [0.005, 0.01, 0.05],
        'lambda_reg': [0.001, 0.05, 0.1],
        'num_hidden': [20, 50, 100, 200],
        'num_hidden_2': [20, 50, 100, 200],
        'n_epochs': 400,
        'validate_params': ["batch_size", "learning_rate", "lambda_reg", 'num_hidden', 'num_hidden_2']}

    # Cross-validate here
    best_params, all_params = validation_helpers.validate_grid_search(train_fn, test_fn,
                                                                      False, train_samples, train_labels, valid_samples,
                                                                      valid_labels, hyperparams, num_repeat=2)

    # Average results due to non-deterministic nature of the model
    f1 = 0
    precision = 0
    recall = 0

    num_repeat = 3

    for i in range(num_repeat):
        model = train_fn(train_labels, train_samples, best_params)
        f1_c, precision_c, recall_c, prediction, _, _, _ = test_fn(valid_labels, valid_samples, model)
        f1 += f1_c
        precision += precision_c
        recall += recall_c

    f1 /= num_repeat
    precision /= num_repeat
    recall /= num_repeat

    print 'AU%d done: prec %.4f, recall %.4f, f1 %.4f\n' % (au, precision, recall, f1)
    print 'All params', all_params
    print 'Best params', best_params

    f.write("%d %.4f %.4f %.4f\n" % (au, precision, recall, f1))

f.close()
