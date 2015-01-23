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
    data_preparation.Prepare_HOG_AU_data_generic_BP4D_dynamic(train_recs, devel_recs, all_aus_bp4d, BP4D_dir, hog_data_dir, pca_loc, geometry=True)

(all_aus_semaine, train_recs, devel_recs, semaine_dir, hog_data_dir) = shared_defs_SEMAINE.shared_defs()

# load the training and testing data for the current fold
[train_samples_semaine, train_labels_semaine, valid_samples_semaine, valid_labels_semaine, _, PC, means, scaling] = \
    data_preparation.Prepare_HOG_AU_data_generic_SEMAINE_dynamic(train_recs, devel_recs, all_aus_semaine,
                                                         semaine_dir, hog_data_dir, pca_loc, geometry=True)

(all_aus_disfa, train_recs, disfa_dir, hog_data_dir) = shared_defs_DISFA.shared_defs()
devel_recs = train_recs[0:1]
[train_samples_disfa, train_labels_disfa, _, _, _, PC, means, scaling] = \
    data_preparation.Prepare_HOG_AU_data_generic_DISFA_dynamic(train_recs, devel_recs, all_aus_disfa,
                                                         disfa_dir, hog_data_dir, pca_loc, geometry=True)

# Binarise disfa labels
train_labels_disfa[train_labels_disfa > 1] = 1

# Train on all three
#  Do the fully joint models first (2, 12, 17)
aus_exp = [2, 12, 17]

inds_to_use_disfa = []
inds_to_use_semaine = []
inds_to_use_bp4d = []

for au in aus_exp:
    inds_to_use_disfa = inds_to_use_disfa + [all_aus_disfa.index(au)]
    inds_to_use_semaine = inds_to_use_semaine + [all_aus_semaine.index(au)]
    inds_to_use_bp4d = inds_to_use_bp4d + [all_aus_bp4d.index(au)]

train_samples = numpy.concatenate((train_samples_bp4d, train_samples_semaine, train_samples_disfa), axis=0)
train_labels = numpy.concatenate((train_labels_bp4d[:, inds_to_use_bp4d],
                                  train_labels_semaine[:, inds_to_use_semaine],
                                  train_labels_disfa[:, inds_to_use_disfa]), axis=0)



#hyperparams = {
#    'batch_size': [100],
#    'learning_rate': [0.02, 0.04, 0.1],
#    'lambda_reg': [0.0001, 0.001, 0.01],
#    'num_hidden': [100, 200, 300, 400],
#    'n_epochs': 100,
#    'validate_params': ["batch_size", "learning_rate", "lambda_reg", 'num_hidden']}

import validation_helpers

train_fn = mlp.train_mlp_probe
test_fn = mlp.test_mlp_class

valid_samples = valid_samples_semaine
valid_labels = valid_labels_semaine[:, inds_to_use_semaine]

print numpy.mean(train_labels, axis=0), numpy.mean(valid_labels, axis=0)

# Cross-validate here
#best_params, all_params = validation_helpers.validate_grid_search_cheat(train_fn, test_fn,
#                                                                  False, train_samples, train_labels, valid_samples,
#                                                                  valid_labels, hyperparams, num_repeat=3)
#print 'All params', all_params

best_params = {
    'batch_size': 100,
    'learning_rate': 0.1,
    'lambda_reg': 0.001,
    'num_hidden': 200,
    'n_epochs': 200,
    'validate_params': ["batch_size", "learning_rate", "lambda_reg", 'num_hidden']}


print 'Best params', best_params

f1_b = 0

for i in range(3):
    model = train_fn(train_labels, train_samples, valid_labels, valid_samples, best_params)
    _, _, _, _, f1, _, _ = test_fn(valid_labels_semaine[:, inds_to_use_semaine], valid_samples_semaine, model)
    if numpy.mean(f1) > f1_b:
        best_model = model
        f1_b = numpy.mean(f1)

model = best_model

model = best_model

# Test on SEMAINE
_, _, _, _, f1s, precisions, recalls = test_fn(valid_labels_semaine[:, inds_to_use_semaine], valid_samples_semaine, model)

f = open("./trained/SEMAINE_train_mlp_combined_geom.txt", 'w')
f.write(str(best_params)+'\n')

for i in range(len(aus_exp)):
    print 'SEMAINE AU%d done: precision %.4f, recall %.4f, f1 %.4f\n' % (aus_exp[i], precisions[i], recalls[i], f1s[i])
    f.write("%d %.4f %.4f %.4f\n" % (aus_exp[i], precisions[i], recalls[i], f1s[i]))

f.close()


# Now the models that only partially intersect, DISFA + SEMAINE - 25
aus_exp = [25]

inds_to_use_disfa = []
inds_to_use_semaine = []

for au in aus_exp:
    inds_to_use_disfa = inds_to_use_disfa + [all_aus_disfa.index(au)]
    inds_to_use_semaine = inds_to_use_semaine + [all_aus_semaine.index(au)]

train_samples = numpy.concatenate((train_samples_semaine, train_samples_disfa), axis=0)
train_labels = numpy.concatenate((train_labels_semaine[:, inds_to_use_semaine],
                                  train_labels_disfa[:, inds_to_use_disfa]), axis=0)

valid_samples = valid_samples_semaine
valid_labels = valid_labels_semaine[:, inds_to_use_semaine]

#hyperparams = {
#    'batch_size': [100],
#    'learning_rate': [0.02, 0.04, 0.1],
#    'lambda_reg': [0.0001, 0.001, 0.01],
#    'num_hidden': [100, 200, 300, 400],
#    'n_epochs': 100,
#    'validate_params': ["batch_size", "learning_rate", "lambda_reg", 'num_hidden']}


print train_samples.shape, train_labels.shape, valid_samples.shape, valid_labels.shape

import validation_helpers

train_fn = mlp.train_mlp_probe
test_fn = mlp.test_mlp_class

# Cross-validate here
#best_params, all_params = validation_helpers.validate_grid_search_cheat(train_fn, test_fn,
#                                                                  False, train_samples, train_labels, valid_samples,
#                                                                  valid_labels, hyperparams, num_repeat=3)
#print 'All params', all_params
print 'Best params', best_params

f1_b = 0

for i in range(3):
    model = train_fn(train_labels, train_samples, valid_labels, valid_samples, best_params)
    _, _, _, _, f1, _, _ = test_fn(valid_labels_semaine[:, inds_to_use_semaine], valid_samples_semaine, model)
    if numpy.mean(f1) > f1_b:
        best_model = model
        f1_b = numpy.mean(f1)

model = best_model

# Test on SEMAINE
_, _, _, _, f1s, precisions, recalls = test_fn(valid_labels_semaine[:, inds_to_use_semaine], valid_samples_semaine, model)

f = open("./trained/SEMAINE_train_mlp_combined_25_geom.txt", 'w')
f.write(str(best_params)+'\n')

for i in range(len(aus_exp)):
    print 'SEMAINE AU%d done: precision %.4f, recall %.4f, f1 %.4f\n' % (aus_exp[i], precisions[i], recalls[i], f1s[i])
    f.write("%d %.4f %.4f %.4f\n" % (aus_exp[i], precisions[i], recalls[i], f1s[i]))

f.close()


