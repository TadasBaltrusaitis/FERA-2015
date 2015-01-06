def FERA_class_score(prediction, ground_truth):
    import numpy as np
    
    tp = np.sum(np.logical_and(ground_truth == 1, prediction == 1));
    fp = np.sum(np.logical_and(ground_truth == 0, prediction == 1));
    fn = np.sum(np.logical_and(ground_truth == 1, prediction == 0));
    tn = np.sum(np.logical_and(ground_truth == 0, prediction == 0));
    
    if(tp+fp != 0):
        precision = float(tp)/float(tp+fp);
    else:
        precision = 0;
            
    if( tp+fn != 0):
        recall = float(tp)/float(tp+fn);
    else:
        recall = 0;
        
    if(precision != 0 or recall != 0):
        f1 = 2 * precision * recall / (precision + recall);    
    else:
        f1 = 0;
        
    return f1, precision, recall