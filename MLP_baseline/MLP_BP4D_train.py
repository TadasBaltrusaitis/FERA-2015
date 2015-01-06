# The MLP baseline for SEMAINE
import shared_defs_BP4D
import data_preparation
(all_aus, train_recs, devel_recs, BP4D_dir, hog_data_dir) = shared_defs_BP4D.shared_defs();

pca_loc = "../pca_generation/generic_face_rigid";

for au in all_aus:
               
    # load the training and testing data for the current fold
    [train_samples, train_labels, valid_samples, valid_labels, raw_valid, PC, means, scaling] = data_preparation.Prepare_HOG_AU_data_generic_BP4D(train_recs, devel_recs, [au], BP4D_dir, hog_data_dir, pca_loc);

    # Cross-validate here                
    #( best_params, _ ) = validate_grid_search(svm_train, svm_test, False, train_samples, train_labels, valid_samples, valid_labels, hyperparams);

    # model = svm_train(train_labels, train_samples, best_params);        

    # [prediction, a, actual_vals] = predict(valid_labels, valid_samples, model);

    # Go from raw data to the prediction
    #w = model.w(1:end-1)';
    #b = model.w(end);

    #svs = bsxfun(@times, PC, 1./scaling') * w;

    # Attempt own prediction
    #preds_mine = bsxfun(@plus, raw_valid, -means) * svs + b;

    # assert(norm(preds_mine - actual_vals) < 1e-8);#

    # name = sprintf('trained/AU_%d_static.dat', au);

    # write_lin_svr(name, means, svs, b);

    # name = sprintf('trained/AU_%d_static.mat', au);

    #tp = sum(valid_labels == 1 & prediction == 1);
    #fp = sum(valid_labels == 0 & prediction == 1);
    #fn = sum(valid_labels == 1 & prediction == 0);
    #tn = sum(valid_labels == 0 & prediction == 0);

    #precision = tp/(tp+fp);
    #recall = tp/(tp+fn);

    #f1 = 2 * precision * recall / (precision + recall);    
    
    #save(name, 'model', 'f1', 'precision', 'recall');
  

