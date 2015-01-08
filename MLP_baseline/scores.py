def FERA_class_score(prediction, ground_truth):
    import numpy as np

    if(len(prediction.shape) == 1):
        prediction.shape = (prediction.shape[0],1)

    if(len(ground_truth.shape) == 1):
        ground_truth.shape = (ground_truth.shape[0],1)

    pos_gt = (ground_truth == 1).astype(int)
    neg_gt = (ground_truth == 0).astype(int)

    pos_pred = (prediction == 1).astype(int)
    neg_pred = (prediction == 0).astype(int)

    tp = np.sum(pos_gt * pos_pred, axis = 0).astype(float)
    fp = np.sum(neg_gt * pos_pred, axis = 0).astype(float)
    fn = np.sum(pos_gt * neg_pred, axis = 0).astype(float)
    tn = np.sum(neg_gt * neg_pred, axis = 0).astype(float)

    precision = tp/(tp+fp)
    precision[tp+fp == 0] = 0
            
    recall = tp/(tp+fn)
    recall[tp+fn == 0] = 0

    f1 = 2 * precision * recall / (precision + recall)

    f1[np.logical_or(precision == 0, recall == 0)] = 0

    return f1, precision, recall