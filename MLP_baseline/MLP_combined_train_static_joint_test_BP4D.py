# The SVM baseline for BO4D
import shared_defs_BP4D
import shared_defs_SEMAINE
import shared_defs_DISFA

import data_preparation
import numpy

import mlp

pca_loc = "../pca_generation/generic_face_rigid"

(all_aus_bp4d, train_recs, devel_recs, BP4D_dir, hog_data_dir) = shared_defs_BP4D.shared_defs()

# load the training and testing data for the current fold
[train_samples_bp4d, train_labels_bp4d, valid_samples_bp4d, valid_labels_bp4d, _, PC, means, scaling] = \
    data_preparation.Prepare_HOG_AU_data_generic_BP4D(train_recs, devel_recs, all_aus_bp4d, BP4D_dir, hog_data_dir, pca_loc)

(all_aus_semaine, train_recs, devel_recs, semaine_dir, hog_data_dir) = shared_defs_SEMAINE.shared_defs()

# load the training and testing data for the current fold
[train_samples_semaine, train_labels_semaine, valid_samples_semaine, valid_labels_semaine, _, PC, means, scaling] = \
    data_preparation.Prepare_HOG_AU_data_generic_SEMAINE(train_recs, devel_recs, all_aus_semaine,
                                                         semaine_dir, hog_data_dir, pca_loc)

(all_aus_disfa, train_recs, disfa_dir, hog_data_dir) = shared_defs_DISFA.shared_defs()
devel_recs = []
[train_samples_disfa, train_labels_disfa, valid_samples_disfa, valid_labels_disfa, _, PC, means, scaling] = \
    data_preparation.Prepare_HOG_AU_data_generic_SEMAINE(train_recs, devel_recs, all_aus_disfa,
                                                         disfa_dir, hog_data_dir, pca_loc)

# Do the fully joint models first (2, 12, 17)
aus_exp = [2, 12, 17]

inds_to_use_disfa = []
inds_to_use_semaine = []
inds_to_use_bp4d = []

for au in aus_exp:
    inds_to_use_disfa = inds_to_use_disfa + [all_aus_disfa.index(au)]
    inds_to_use_semaine = inds_to_use_semaine + [all_aus_semaine.index(au)]
    inds_to_use_bp4d = inds_to_use_bp4d + [all_aus_bp4d.index(au)]

# Test on BP4D
train_samples = numpy.concatenate((train_samples_bp4d, train_samples_semaine, train_samples_disfa), axis=0)
train_labels = numpy.concatenate((train_labels_bp4d[:, inds_to_use_bp4d],
                                  train_labels_semaine[:, inds_to_use_semaine],
                                  train_samples_disfa[:, inds_to_use_disfa]), axis=0)

valid_samples = numpy.concatenate((valid_samples_bp4d, valid_samples_semaine), axis=0)
valid_labels = numpy.concatenate((valid_labels_bp4d[:, inds_to_use_bp4d],
                                  valid_labels_semaine[:, inds_to_use_semaine]), axis=0)
hyperparams = {
   'batch_size': [100],
   'learning_rate': [0.05, 0.1],
   'lambda_reg': [0.001, 0.005],
   'num_hidden': [50, 100, 200],
   'n_epochs': 20,
   'validate_params': ["batch_size", "learning_rate", "lambda_reg", 'num_hidden']}

import validation_helpers

train_fn = mlp.train_mlp
test_fn = mlp.test_mlp

# Cross-validate here
best_params, all_params = validation_helpers.validate_grid_search(train_fn, test_fn,
                                                                  False, train_samples, train_labels, valid_samples,
                                                                  valid_labels, hyperparams, num_repeat=1)
print 'All params', all_params
print 'Best params', best_params

model = train_fn(train_labels, train_samples, best_params)

# Test on SEMAINE
_, _, _, _, f1s, precisions, recalls = test_fn(valid_labels_semaine, valid_samples_semaine, model)

f = open("./trained/SEMAINE_train_mlp_combined.txt", 'w')

for i in range(len(aus_exp)):
    print 'SEMAINE AU%d done: precision %.4f, recall %.4f, f1 %.4f\n' % (aus_exp[i], precisions[0, i], recalls[0, i], f1s[0, i])
    f.write("%d %.4f %.4f %.4f\n" % (aus_exp[i], precisions[0, i], recalls[0, i], f1s[0, i]))

f.close()

# Test on SEMAINE
_, _, _, _, f1s, precisions, recalls = test_fn(valid_labels_bp4d, valid_samples_bp4d, model)

f = open("./trained/SEMAINE_train_mlp_combined.txt", 'w')

for i in range(len(aus_exp)):
    print 'SEMAINE AU%d done: precision %.4f, recall %.4f, f1 %.4f\n' % (aus_exp[i], precisions[0, i], recalls[0, i], f1s[0, i])
    f.write("%d %.4f %.4f %.4f\n" % (aus_exp[i], precisions[0, i], recalls[0, i], f1s[0, i]))

f.close()

# Now the models that only partially intersect, DISFA + SEMAINE - 25

# DISFA + BP4D - 1, 4, 6, 15

# SEMAINE + BP4D - 2, 12, 17 (basically without DISFA)


