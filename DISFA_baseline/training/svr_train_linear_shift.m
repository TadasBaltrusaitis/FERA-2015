function [model] = svr_train_linear_shift(train_labels, train_samples, hyper)
    % Change to your downloaded location
    addpath('C:\liblinear\matlab')
    
    comm = sprintf('-s 11 -B 1 -p %.10f -c %.10f -q', hyper.p, hyper.c);
    model = train(train_labels, train_samples, comm);
    model.vid_ids = hyper.vid_ids;
end