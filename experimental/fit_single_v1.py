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
'''
argparser = argparse.ArgumentParser()
argparser.add_argument('--subject_id', type=int)
argparser.add_argument('--max_iter', type=int)
argparser.add_argument('--file_name_prefix', type=str, default='opt_out_subject_id_')
args = argparser.parse_args()

fixed_params = {}

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

def scale_and_bias(x,bias=0,scale=45.0):
    x = np.array(x)
    return x*scale + bias


optim_param_mapping= [('runner','learning_rate',lambda x: 10.**x), 
                      ('model','skip_gain'), 
                      ('model','nl',
                       lambda w: (lambda : OneOverSqr(w=w))),
                     ('postprocessing','scale')]

wrapped_runner = wrap_runner_for_optimization(model_class=MLP, 
                                             fixed_params=fixed_params,
                                             optim_params_mapping=optim_param_mapping,
                                             postprocessing_fun=scale_and_bias)

df = pd.read_csv('../../../frogs_project/data/avraham__ivry_fig1_rotFig1_v2.csv',header=None)
df = df.T
data = df.to_numpy()
this_data = data[:80,args.subject_id]

stimulus = [(1,40),(0,40)]
pooling_funs = ['MSE']


fitting_loss = {pooling_fun : create_fitting_loss(data=this_data,stimulus=stimulus,wrapped_model=wrapped_runner,
                    pooling_fun=pooling_fun,datapoint_mapping=None,weighting=None) for pooling_fun in pooling_funs}


x0 = [-4.5,0.4,0.5, 40]
bounds = [(-6,-3),(-0.9,0.99), (0.05,3), (10,90)]


out = wrapped_runner([(0,40),(1,40)]*3,x0)
opt_out = {}

def print_fun(x, f, accepted):
        print("at minimum %.4f accepted %d  with params %s" % (f, int(accepted), str(x)))

for pooling_fun in pooling_funs:
    opt_out[pooling_fun] = basinhopping(fitting_loss[pooling_fun], x0, niter=args.max_iter,
                    minimizer_kwargs = dict(method='L-BFGS-B', bounds= bounds),
                    callback=print_fun)

with open(f'{args.file_name_prefix}{args.subject_id}.pkl','wb') as f:
    pickle.dump(opt_out,f)
