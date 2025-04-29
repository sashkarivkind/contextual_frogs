'''
v2, added support for fitting coin data
'''

# %%
import sys

# Specify the directory you want to add
custom_path = './..'

# Add the directory to sys.path
if custom_path not in sys.path:
    sys.path.append(custom_path)




import numpy as np
import pandas as pd
from scipy.optimize import minimize, basinhopping



from models import MLP, OneOverSqr
from runners import wrap_runner_for_optimization
from fitting_utils import create_fitting_loss

import pickle

import argparse


'''
parser arguments
--subject_id
--max_iter
--file_name_prefix default = 'opt_out_subject_id_'
--param_config_id - integer, selecting the parameter configuration to use
--shift_model_out - if enabled, model output is extended by one timesteps and then shifted left by one step 
--fitting_loss - loss function to use for parameter fitting, the default is MSE.
'''
argparser = argparse.ArgumentParser()
argparser.add_argument('--subject_id', type=int)
argparser.add_argument('--max_iter', type=int)
argparser.add_argument('--experimental_data', type=str, default='avraham')
argparser.add_argument('--paradigm', type=str, default=None)
argparser.add_argument('--file_name_prefix', type=str, default='opt_out_subject_id_')
argparser.add_argument('--param_config_id', type=int, default=1)
argparser.add_argument('--shift_model_out', action='store_true')
argparser.add_argument('--fitting_loss', type=str, default='MSE')
argparser.set_defaults(shift_model_out=False)
args = argparser.parse_args()


def scale_and_bias(x,bias=0,scale=45.0):
    x = np.array(x)
    return x*scale + bias

x0 = [-4.5,0.4,0.5, 40]
bounds = [(-6,-3),(-0.9,0.99), (0.05,3), (10,90)]

fixed_params = {}
custom_param_mappings = []


if args.param_config_id == 1:
    fixed_params['model'] =  dict(n_inputs = 3,
                    n_hidden = 5*4*512,
                    n_outs = 1,
                    en_bias = False,
                    b_high=3, first_layer_init='ones',
                    first_layer_weights_trainable = True,
                    out_layer_init='zeros')         
    #todo - tie "kernel width" to learning rate 
    fixed_params['runner'] = {'criterion':'MSE', 'k':[0,1,0], 'sigma_noi':0.0, 'tau_u':1,
                            'save_model_at_init':False, 'ic_param_file':None}

    optim_param_mapping= [('runner','learning_rate',lambda x: 10.**x), 
                        ('model','skip_gain'), 
                        ('model','nl',
                        lambda w: (lambda : OneOverSqr(w=w))),
                        ('postprocessing','scale')]

elif args.param_config_id == 2:
    fixed_params['model'] =  dict(n_inputs = 3,
            n_hidden = 5*4*512,
            n_outs = 1,
            en_bias = False,
            b_high=3, first_layer_init='ones',
            first_layer_weights_trainable = True,
            out_layer_init='zeros')         

    fixed_params['runner'] = {'criterion':'MSE', 'k':[0,1,0], 'sigma_noi':0.0, 'tau_u':1,
                            'save_model_at_init':False, 'ic_param_file':None}

    optim_param_mapping= [('custom','normalized_log_lr'),
                        ('model','skip_gain'),                      
                        ('custom','w'),
                        ('postprocessing','scale'),
                        ]

    custom_param_mappings = [{'cathegory':'runner','param_name':'learning_rate',
                            'fun': lambda x: 10.**x['normalized_log_lr']/x['w']},
                            {'cathegory':'model','param_name':'nl','fun': lambda x: (lambda : OneOverSqr(w=x['w']))}]
    
elif args.param_config_id == 3:
    x0 = [-4.5,0.4,0.5, 40, 1, 1]
    bounds = [(-6,-3),(-0.9,0.99), (0.05,3), (10,90), (0,2), (0,2)]

    fixed_params['model'] =  dict(n_inputs = 3,
            n_hidden = 5*4*512,
            n_outs = 1,
            en_bias = False,
            b_high=3, first_layer_init='uniform_unity',
            first_layer_weights_trainable = True,
            out_layer_init='zeros')         

    fixed_params['runner'] = {'criterion':'MSE', 'k':[0,1,0], 'sigma_noi':0.0, 'tau_u':1,
                            'save_model_at_init':False, 'ic_param_file':None}

    optim_param_mapping= [('custom','normalized_log_lr'),
                        ('model','skip_gain'),                      
                        ('custom','w'),
                        ('postprocessing','scale'),
                        ('custom', 'k1'),
                        ('custom', 'k3'),
                        ]

    custom_param_mappings = [{'cathegory':'runner','param_name':'learning_rate',
                            'fun': lambda x: 10.**x['normalized_log_lr']/x['w']},
                            {'cathegory':'runner','param_name':'k',
                            'fun': lambda x: [x['k1'],0,x['k3']]},
                            {'cathegory':'model','param_name':'nl','fun': lambda x: (lambda : OneOverSqr(w=x['w']))}]

elif args.param_config_id == 4:

    x0 = [-4.5,0.4,0.05, 1.0]
    bounds = [(-6,-3),(-0.1,0.99), (0.01,0.5), (0.5,1.1)]

    fixed_params['model'] =  dict(n_inputs = 4,
                          n_hidden = 5*4*512,
                          n_outs = 1,
                          en_bias = False,
                         first_layer_init='uniform_unity',
                        first_layer_weights_trainable = True,
                        out_layer_init='zeros',
                          nl = 'relu')        

    fixed_params['runner'] = {'criterion':'MSE', 'k':[0,0,0,1],  'sigma_noi':0.0, 'tau_u':1,
                            'save_model_at_init':False, 'ic_param_file':None, 'enable_combo':True}

    optim_param_mapping= [('custom','normalized_log_lr'),
                        ('model','skip_gain'),                      
                        ('model','b_high'),                      
                        ('postprocessing','scale'),
                        ]
    custom_param_mappings = [{'cathegory':'runner','param_name':'learning_rate',
                            'fun': lambda x: 10.**x['normalized_log_lr']},]

elif args.param_config_id == 5:

    x0 = [-4.5,0.4,0.05, 40]
    bounds = [(-6,-3),(-0.1,0.99), (0.01,2), (10,90)]

    fixed_params['model'] =  dict(n_inputs = 4,
                          n_hidden = 5*4*512,
                          n_outs = 1,
                          en_bias = False,
                         first_layer_init='uniform_unity',
                        first_layer_weights_trainable = True,
                        out_layer_init='zeros',
                          nl = 'relu')        

    fixed_params['runner'] = {'criterion':'MSE', 'k':[0,0,0,1],  'sigma_noi':0.0, 'tau_u':1,
                            'save_model_at_init':False, 'ic_param_file':None, 'enable_combo':True}

    optim_param_mapping= [('custom','normalized_log_lr'),
                        ('model','skip_gain'),                      
                        ('model','b_high'),                      
                        ('postprocessing','scale'),
                        ]
    custom_param_mappings = [{'cathegory':'runner','param_name':'learning_rate',
                            'fun': lambda x: 10.**x['normalized_log_lr']},]

else:
    raise ValueError('param_config_id not recognized')


wrapped_runner = wrap_runner_for_optimization(model_class=MLP, 
                                             fixed_params=fixed_params,
                                             optim_params_mapping=optim_param_mapping,
                                             custom_param_mappings=custom_param_mappings, 
                                             postprocessing_fun=scale_and_bias)

if args.experimental_data == 'avraham':
    df = pd.read_csv('../../../frogs_project/data/avraham__ivry_fig1_rotFig1_v2.csv',header=None)
    df = df.T
    data = df.to_numpy()
    this_data = data[:80,args.subject_id]

    if args.shift_model_out:
        stimulus = [(1,40),(0,41)]    
        datapoint_mapping={'data':lambda x:x, 'model_output': lambda x:x[1:]} #a hack that mitigates the fact that first perturbatrion datapoint is 'responsive'
    else:
        stimulus = [(1,40),(0,40)]
        datapoint_mapping=None

elif args.experimental_data == 'coin':
    Pplus = 1
    Pminus = -1
    P0 = 0
    Pchannel = np.nan

    '''
    spontaneous:
    trials in block Null: 50
    trials in block FieldA: 120
    trials in block PostRest: 5
    trials in block FieldB: 15
    trials in block Clamp150: 150


    evoked:
    trials in block Null: 50
    trials in block FieldA: 120
    trials in block PostRest: 5
    trials in block FieldB: 15
    trials in block Clamp2: 2
    trials in block FieldA2: 2
    trials in block Clamp150: 146
    '''
    stimuli = {'spontaneous': [(P0, 50),
                            (Pplus, 125),
                            (Pminus, 15),
                            (Pchannel, 150)],
                'evoked': [(P0, 50),
                            (Pplus, 125),
                            (Pminus, 15),
                            (Pchannel, 2),
                            (Pplus, 2),
                            (Pchannel, 146)]}

    df = pd.read_csv(f'../../../frogs_project/data/COIN_data/trial_data_{args.paradigm}_recovery_participant{args.subject_id}.csv')
    y = df.Adaptation.to_numpy()
    y *= np.sign(np.nansum(y))
    this_data = y
    stimulus = stimuli[args.paradigm]
    datapoint_mapping=None
    if args.shift_model_out:
        raise NotImplementedError('shift_model_out not implemented for coin data')


pooling_funs = [args.fitting_loss]


fitting_loss = {pooling_fun : create_fitting_loss(data=this_data,stimulus=stimulus,wrapped_model=wrapped_runner,
                    pooling_fun=pooling_fun,datapoint_mapping=datapoint_mapping,weighting=None) for pooling_fun in pooling_funs}

def print_fun(x, f, accepted):
        print("at minimum %.4f accepted %d  with params %s" % (f, int(accepted), str(x)))

opt_out = {}
for pooling_fun in pooling_funs:
    opt_out[pooling_fun] = basinhopping(fitting_loss[pooling_fun], x0, niter=args.max_iter,
                    minimizer_kwargs = dict(method='L-BFGS-B', bounds= bounds),
                    callback=print_fun)

with open(f'{args.file_name_prefix}{args.subject_id}.pkl','wb') as f:
    pickle.dump(opt_out,f)

with open(f'{args.file_name_prefix}{args.subject_id}.cfg','wb') as f:
    pickle.dump(args,f)