# The SVM baseline for BO4D
import shared_defs_BP4D
import data_preparation
import numpy

import mlp

(all_aus, train_recs, devel_recs, BP4D_dir, hog_data_dir) = shared_defs_BP4D.shared_defs_intensity()

pca_loc = "../pca_generation/generic_face_rigid"


# load the training and testing data for the current fold
[train_samples, train_labels, valid_samples, valid_labels, raw_valid, PC, means, scaling] = \
    data_preparation.Prepare_HOG_AU_data_generic_BP4D_intensity(train_recs, devel_recs, all_aus, BP4D_dir, hog_data_dir, pca_loc, geometry=True)

import validation_helpers

train_labels = train_labels / 5.0
valid_labels = valid_labels / 5.0

train_fn = mlp.train_mlp_probe
test_fn = mlp.test_mlp_reg

hyperparams = {
   'batch_size': [100],
   'learning_rate': [0.8, 2.0],
   'lambda_reg': [0.004],
   'num_hidden': [50],
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

f = open("./trained/BP4D_train_mlp_joint_intensity_geom.txt", 'w')
f.write(str(best_params)+'\n')

for i in range(len(all_aus)):
    print 'AU%d done: correlation %.4f, MSE %.4f \n' % (all_aus[i], corrs[0, i], mses[0, i])
    f.write("%d %.4f %.4f\n" % (all_aus[i], corrs[0, i], mses[0, i]))

f.close()
