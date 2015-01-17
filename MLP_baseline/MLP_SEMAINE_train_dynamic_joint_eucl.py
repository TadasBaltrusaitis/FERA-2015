# The SVM baseline for SEMAINE
import shared_defs_SEMAINE
import data_preparation
import numpy

import mlp

(all_aus, train_recs, devel_recs, SEMAINE_dir, hog_data_dir) = shared_defs_SEMAINE.shared_defs()

pca_loc = "../pca_generation/generic_face_rigid"


# load the training and testing data for the current fold
[train_samples, train_labels, valid_samples, valid_labels, raw_valid, PC, means, scaling] = \
    data_preparation.Prepare_HOG_AU_data_generic_SEMAINE_dynamic(train_recs, devel_recs, all_aus, SEMAINE_dir,
                                                                 hog_data_dir, pca_loc, scale=False)

import validation_helpers

train_fn = mlp.train_mlp_probe
test_fn = mlp.test_mlp_class

hyperparams = {
   'batch_size': [100],
   'learning_rate': [0.1, 0.2],
   'lambda_reg': [0.00001, 0.0001, 0.001],
   'num_hidden': [250],
   'n_epochs': 1000,
   'error_func': 'euclidean',
   'validate_params': ["batch_size", "learning_rate", "lambda_reg", 'num_hidden']}

# Cross-validate here
best_params, all_params = validation_helpers.validate_grid_search_cheat(train_fn, test_fn,
                                                                  False, train_samples, train_labels, valid_samples,
                                                                  valid_labels, hyperparams, num_repeat=2)

# Average results due to non-deterministic nature of the model
f1s = numpy.zeros((1, train_labels.shape[1]))
precisions = numpy.zeros((1, train_labels.shape[1]))
recalls = numpy.zeros((1, train_labels.shape[1]))

num_repeat = 3

print 'All params', all_params
print 'Best params', best_params

for i in range(num_repeat):
    model = train_fn(train_labels, train_samples, valid_labels, valid_samples, best_params)
    _, _, _, _, f1_c, precision_c, recall_c = test_fn(valid_labels, valid_samples, model)
    f1s += f1_c
    precisions += precision_c
    recalls += recall_c

f1s /= num_repeat
precisions /= num_repeat
recalls /= num_repeat

f = open("./trained/SEMAINE_train_mlp_joint_dynami_eucl.txt", 'w')
f.write(str(best_params) + '\n')
for i in range(len(all_aus)):
    print 'AU%d done: precision %.4f, recall %.4f, f1 %.4f\n' % (all_aus[i], precisions[0, i], recalls[0, i], f1s[0, i])
    f.write("%d %.4f %.4f %.4f\n" % (all_aus[i], precisions[0, i], recalls[0, i], f1s[0, i]))

f.close()
