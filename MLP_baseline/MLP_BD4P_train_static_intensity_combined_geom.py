# The SVM baseline for BO4D
import shared_defs_BP4D
import shared_defs_DISFA
import data_preparation
import numpy

import mlp

(all_aus_bp4d, train_recs, devel_recs, BP4D_dir, hog_data_dir) = shared_defs_BP4D.shared_defs_intensity()

pca_loc = "../pca_generation/generic_face_rigid"


# load the training and testing data for the current fold
[train_samples_bp4d, train_labels_bp4d, valid_samples_bp4d, valid_labels_bp4d, raw_valid, PC, means, scaling] = \
    data_preparation.Prepare_HOG_AU_data_generic_BP4D_intensity(train_recs, devel_recs, all_aus_bp4d, BP4D_dir, hog_data_dir, pca_loc, geometry=True)

(all_aus_disfa, train_recs, disfa_dir, hog_data_dir) = shared_defs_DISFA.shared_defs()
devel_recs = train_recs[0:1]
[train_samples_disfa, train_labels_disfa, _, _, _, PC, means, scaling] = \
    data_preparation.Prepare_HOG_AU_data_generic_DISFA(train_recs, devel_recs, all_aus_disfa,
                                                         disfa_dir, hog_data_dir, pca_loc, geometry=True)

import validation_helpers

train_labels_bp4d = train_labels_bp4d / 5.0
valid_labels_bp4d = valid_labels_bp4d / 5.0

train_labels_disfa = train_labels_disfa / 5.0

# Train on all three
#  Do the fully joint models first (2, 12, 17)
aus_exp = [6, 12, 17]

inds_to_use_disfa = []
inds_to_use_bp4d = []

for au in aus_exp:
    inds_to_use_disfa = inds_to_use_disfa + [all_aus_disfa.index(au)]
    inds_to_use_bp4d = inds_to_use_bp4d + [all_aus_bp4d.index(au)]

train_samples = numpy.concatenate((train_samples_bp4d, train_samples_disfa), axis=0)
train_labels = numpy.concatenate((train_labels_bp4d[:, inds_to_use_bp4d],
                                  train_labels_disfa[:, inds_to_use_disfa]), axis=0)

valid_samples = valid_samples_bp4d
valid_labels = valid_labels_bp4d[:, inds_to_use_bp4d]

train_fn = mlp.train_mlp_probe
test_fn = mlp.test_mlp_reg

hyperparams = {
   'batch_size': [100],
   'learning_rate': [0.8],
   'lambda_reg': [0.001],
   'num_hidden': [50, 100, 200],
   'final_layer': ['sigmoid'],
   'error_func': 'euclidean',
   'pred_type': 'reg',
   'n_epochs': 1000,
   'validate_params': ["batch_size", "learning_rate", "lambda_reg", 'num_hidden', 'final_layer']}

# Cross-validate here
best_params, all_params = validation_helpers.validate_grid_search_cheat(train_fn, test_fn,
                                                                  False, train_samples, train_labels, valid_samples,
                                                                  valid_labels, hyperparams, num_repeat=2)

# Average results due to non-deterministic nature of the model
corrs = numpy.zeros((1, train_labels.shape[1]))
mses = numpy.zeros((1, train_labels.shape[1]))

num_repeat = 1

print 'All params', all_params
print 'Best params', best_params

for i in range(num_repeat):
    model = train_fn(train_labels, train_samples, valid_labels, valid_samples, best_params)
    _, _, _, corr, mse = test_fn(valid_labels, valid_samples, model)
    corrs += corr
    mses += mse

corrs /= num_repeat
mses /= num_repeat

f = open("./trained/BP4D_train_mlp_combined_intensity_geom.txt", 'w')
f.write(str(best_params)+'\n')

for i in range(len(aus_exp)):
    print 'AU%d done: correlation %.4f, MSE %.4f \n' % (aus_exp[i], corrs[0, i], mses[0, i])
    f.write("%d %.4f %.4f\n" % (aus_exp[i], corrs[0, i], mses[0, i]))

f.close()
