import numpy as np
import types

def create_fitting_loss(data=None,stimulus=None,wrapped_model=None,pooling_fun='MSE',datapoint_mapping=None,weighting=None, nan_replacement=1e20):

    '''
    Create a loss function for fitting a model to data.
    
    Parameters:
    data: array-like
        The data to fit to.
    stimulus: array-like
        The stimulus to use for the model.
    wrapped_model: function
        The model to fit to the data. The model should take the stimulus and a parameter array as input.
    pooling_fun: function or string
        The function to use to pool the errors across the data.
    datapoint_mapping: dict
        A dictionary with keys 'data' and 'model_output'. 
        The values should be functions that take the data and model output, respectively, and return the data and model output in the correct format for the loss function.
    weighting: function
        A function to apply to the errors before pooling.
    nan_replacement: float or None
        The value to replace NaNs in the loss function (important because optimizers sometimes assume nan < x for any float x).
    '''

    if pooling_fun == 'MSE':
        pooling_fun = lambda x,y: np.nanmean((x-y)**2)
    elif pooling_fun == 'MAE':
        pooling_fun = lambda x,y: np.nanmean(np.abs(x-y))
    elif isinstance(pooling_fun,types.FunctionType):
        pass
    else:
        raise ValueError('pooling_fun must be a Function or one of the supported strings ("MSE"/"MAE).')
    
    if datapoint_mapping is not None:
        data_ = datapoint_mapping['data'](data)
    else:
        data_ = data

    def loss_fun(params):

        model_output = wrapped_model(stimulus,params)
        model_output_ = datapoint_mapping['model_output'](model_output) if datapoint_mapping is not None else model_output
        
        # errors = data_ - model_output_

        if weighting is not None:
            raise NotImplementedError('Weighting is not yet implemented.')
            # errors = weighting(errors)

        loss = pooling_fun(data_, model_output_)

        if np.isnan(loss) and nan_replacement is not None:
            loss = nan_replacement

        return loss
    
    return loss_fun

    