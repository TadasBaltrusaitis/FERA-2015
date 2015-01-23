# The SVM baseline for SEMAINE
import shared_defs_DISFA
import data_preparation
import numpy

import mlp

pca_loc = "../pca_generation/generic_face_rigid"

(all_aus, train_recs, disfa_dir, hog_data_dir) = shared_defs_DISFA.shared_defs()
devel_recs = train_recs[14:-1]
train_recs = train_recs[0:14]

[train_samples, train_labels, valid_samples, valid_labels, _, PC, means, scaling] = \
    data_preparation.Prepare_HOG_AU_data_generic_DISFA(train_recs, devel_recs, all_aus,
                                                         disfa_dir, hog_data_dir, pca_loc, geometry=True)

# binarise the labels
train_labels[train_labels > 1] = 1
valid_labels[valid_labels > 1] = 1

print train_samples.shape, valid_samples.shape, numpy.mean(train_labels, axis=0), numpy.mean(valid_labels, axis=0)


import validation_helpers

train_fn = mlp.train_mlp_probe
test_fn = mlp.test_mlp_class

hyperparams = {
    'batch_size': [100],
    'learning_rate': [0.2],
    'lambda_reg': [0.000001],
    'num_hidden': [50, 100, 200],
    'n_epochs': 50,
    'validate_params': ["batch_size", "learning_rate", "lambda_reg", 'num_hidden']}

# Cross-validate here
best_params, all_params = validation_helpers.validate_grid_search_cheat(train_fn, test_fn,
                                                                  False, train_samples, train_labels, valid_samples,
                                                                  valid_labels, hyperparams, num_repeat=3)

# Average results due to non-deterministic nature of the model
f1s = numpy.zeros((1, train_labels.shape[1]))
precisions = numpy.zeros((1, train_labels.shape[1]))
recalls = numpy.zeros((1, train_labels.shape[1]))

num_repeat = 3

print 'All params', all_params
print 'Best params', best_params

best_params['n_epochs'] = 300

f1_b = 0

for i in range(3):
    model = train_fn(train_labels, train_samples, valid_labels, valid_samples, best_params)
    _, _, _, _, f1, _, _ = test_fn(valid_labels, valid_samples, model)
    if numpy.mean(f1) > f1_b:
        best_model = model
        f1_b = numpy.mean(f1)

model = best_model

# Test on SEMAINE
_, _, _, _, f1s, precisions, recalls = test_fn(valid_labels, valid_samples, model)

f = open("./trained/DISFA_train_mlp_combined_static_geom.txt", 'w')
f.write(str(best_params)+'\n')

for i in range(len(all_aus)):
    print 'DISFA AU%d done: precision %.4f, recall %.4f, f1 %.4f\n' % (all_aus[i], precisions[i], recalls[i], f1s[i])
    f.write("%d %.4f %.4f %.4f\n" % (all_aus[i], precisions[i], recalls[i], f1s[i]))

f.close()
