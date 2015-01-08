def FERA_class_score(prediction, ground_truth):
    import numpy as np

    pos_gt = ground_truth == 1
    neg_gt = ground_truth == 0

    pos_pred = prediction == 1
    neg_pred = prediction == 0

    tp = np.sum(np.logical_and(pos_gt, pos_pred), axis = 0).astype(float)
    fp = np.sum(np.logical_and(neg_gt, pos_pred), axis = 0).astype(float)
    fn = np.sum(np.logical_and(pos_gt, neg_pred), axis = 0).astype(float)
    tn = np.sum(np.logical_and(neg_gt, neg_pred), axis = 0).astype(float)

    precision = tp/(tp+fp)
    precision[tp+fp == 0] = 0
            
    recall = tp/(tp+fn)
    recall[tp+fn == 0] = 0

    f1 = 2 * precision * recall / (precision + recall)

    f1[np.logical_or(precision == 0, recall == 0)] = 0

    return f1, precision, recall