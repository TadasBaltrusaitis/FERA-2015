def validate_grid_search(train_fn, test_fn, minimise, samples_train, labels_train, samples_valid, labels_valid, hyperparams, num_repeat=1):
    #crossvalidate_regressor_grid_search A utility function for crossvalidating a statistical model
    #Detailed explanation goes here

    # train_fn - a function handle that takes train_labels, train_samples, hyperparams as input (with each row being a sample), it must return a model that can be passed to test_fn

    # test_fn - a function that takes test_labels, test_samples, model as 
    # input and returns the result to optimise
 
    # minimise - if set to true the crossvalidation will attempt to find
    # hyper-parameters that minimise the result otherwise they will maximise it

    # samples - the whole training dataset (rows are samples)

    # labels - the labels for training (rows are samples)

    #   hyperparams - the field validate_params should contain the names of 
    #   hyperparameters to validate, and the hyperparameter to be validated 
    #   should contain values to be tested. For example:
    #   If we havehyperparams.validate_params = {'c','g'}, and 
    #   hyperparams.c = [0.1, 10, 100], hyperparams.g = [0.25, 0.5], the grid 
    #   search algorithm will search through all their possible combinations
    #
    #   Optional parameters:
    #
    #   'num_repeat' - number of times to retry the training testing (useful
    #   for non deterministic algorithms

    # Find the hyperparameters to optimise (if any)

    import numpy as np
    
    num_params = 1
    
    if 'validate_params' in hyperparams:
        param_names = hyperparams['validate_params']
        param_values = []
    
        for p in param_names:
            param_values.append(hyperparams[p])
            num_params = num_params * len(hyperparams[p])
    
        # Create the list of parameter combinations          
    
        # keep track of parameter value indices (will be cycling over them based on change_every)
        index = np.zeros((len(param_values)), 'int32')
        change_every = np.zeros((len(param_values)), 'int32')
    
        change_already = num_params
    
        for p in range(len(param_names)):
            change_every[p] = change_already / len(param_values[p])
            change_already /= len(param_values[p])
        
        all_params = []
    
        for i in range(num_params):
            copied_params = hyperparams.copy();
    
            # Some cleanup
            copied_params.pop('validate_params')
    
            all_params.append(copied_params)
    
            for p in range(len(param_names)):
    
                all_params[i][param_names[p]] = param_values[p][index[p]]
    
                # get the new value
                if (i + 1) % change_every[p] == 0:
                    index[p] += 1
    
                # cycle the value if it exceeds the bounds
                if (index[p] % len(param_values[p])) == 0:
                    index[p] = 0
                            
        # Initialise all results to 0
        for i in range(num_params):
            all_params[i]["result"] = 0            
    
    else:
        # if no validation needed just set to hyperparams
        all_params = [hyperparams.copy()]
        all_params[0]["result"] = 0

    print all_params

    # Crossvalidate the c, p, and gamma values
    for p in range(num_params):
        all_params[p]["result"] = single_pass(train_fn, test_fn, labels_train, samples_train, labels_valid,
                                              samples_valid, all_params[p], num_repeat)
        print all_params[p]

    # Finding the best hyper-params
    if minimise:
        results = np.array([item["result"] for item in all_params])
        best = results.argmin()
    else:
        results = np.array([item["result"] for item in all_params])
        best = results.argmax()
    
    best_params = all_params[best]
    
    return best_params, all_params


def single_pass(train_fn, test_fn, labels_train, samples_train, labels_valid, samples_valid, hyperparams, num_repeat):
    result = 0

    for r in range(num_repeat):
        model = train_fn(labels_train, samples_train, hyperparams)
        result += test_fn(labels_valid, samples_valid, model)[0]
    result = result / num_repeat
    return result