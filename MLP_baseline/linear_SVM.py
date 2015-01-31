def train_SVM(train_labels, train_samples, hyperparams):
    
    from sklearn import svm
    clf = svm.LinearSVC(C=hyperparams['C'])
    clf.fit(train_samples, train_labels)
    
    return clf
    
def test_SVM(test_labels, test_samples, model):
    import numpy as np
    import scores
    
    preds = np.transpose(model.predict(test_samples))

    preds = preds.astype('int32');
    test_l = test_labels.astype('int32');
    test_l = test_l[:,0]

    f1, precision, recall = scores.FERA_class_score(preds, test_l)
    
    return f1, precision, recall, preds

def train_SVM_weights(train_labels, train_samples, hyperparams):

    from sklearn import svm
    clf = svm.LinearSVC(C=hyperparams['C'], class_weight='auto')
    clf.fit(train_samples, train_labels)

    return clf