import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import copy
from types import SimpleNamespace

from models import MLP
from dsp_utils import LPF, parse_samples


class Runner:
    def __init__(self,
                model=None,
                model_class=MLP,
                model_construct_args=None,
                rnn_mode = False,
                optimizer_class=None,
                optimizer_opts={},
                learning_rate = 1e-5,
                criterion='MSE',
                device=None,
                ic_param_file='model_parameters.pth',
                create_ic_param_file=True,
                step_method_alias = 'step_vanilla',
                tau_u=1.0,
                loud=True,
                model_type='torch',
                runner_method_alias = 'step_by_step',
                do_backprop=True,
                k=[0,1.,0],
                constancy_factor=None,
                enable_combo=False,
                sigma_noi = 0,
                test_vec=None,
                initial_state=None,
                apply_initial_state=True,
                load_model_at_init=False,
                save_model_at_init=True,
                fb_on_nan=lambda x,y:0.,
                auto_steps=0,
                grad_less_steps=0,
                info=None, #not in use
                ):
        """
        Initialize the Runner class.

        Args:
            model: The machine learning model (PyTorch or NumPy-based). By default is none and instead the model_class is used.
            model_class: Class of the modelc constructor to be used (default is MLP).
            model_construct_args (dict): Arguments for constructing the model.
            optimizer_class: Class of the optimizer to be used (keep None for default SGD).
            optimizer_opts: Options for the optimizer.
            learning_rate: Learning rate for the optimizer.
            criterion: Loss function used for training, or a known string, defaul MSE.
            device (str): Device to use for pytorch. None to use default.
            ic_param_file (str): Path to the file with initial model parameters.
            create_ic_param_file (bool): [todo; Not implemented] If True, creates a new file for initial model parameters.
            tau_u (float): Time constant for the low-pass filter.
            loud (bool): If True, prints reset messages.
            model_type (str): Type of model, either 'torch' or 'numpy'.
            runner_method_alias (str): Alias for the runner method, either 'step_by_step' or 'blackbox'.
            do_backprop (bool): Whether to perform backpropagation during training.
            k (array of float): Coefficients for model input scaling.
            constancy_factor (float): Factor for the constancy in the training. If not None, the training objective y is updated to: 
                (1-constancy_factor)*y + constancy_factor*u_tm1.
            enable_combo (bool): If True, enables the combo u+e.
            sigma_noi (float): Noise level for the model. Not implemented yet.
            test_vec (array): Test vector for evaluation.
            initial_state (array): Initial state of the system.
            apply_initial_state (bool): If True, applies the initial state.
            load_model_at_init (bool): If True, loads the model parameters from the file at initialization.
            save_model_at_init (bool): If True, saves the model parameters to the file at initialization.
            fb_on_nan (function): Function to handle NaN values in the feedback.
            auto_steps (int): Number of underhood automatic steps to take after each recorded step.
            grad_less_steps (int): Number of steps to take without gradient updates after each recorded step.
            info (dict): Additional information (not used); for interface consistency.
        """
        if np.abs(sigma_noi)>1e-20:
            raise NotImplementedError
        self.optimizer_class = optimizer_class if optimizer_class is not None else optim.SGD
        self.optimizer_opts = optimizer_opts
        self.criterion = criterion
        if model_type == 'torch':
            self.parse_criterion()
        
        self.model = model_class(**model_construct_args) if model is None else model            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device    

        if apply_initial_state:   
            self.initial_state = np.zeros(model_construct_args['n_inputs']) if initial_state is None else initial_state
        else:
            self.initial_state = None
            
        self.ic_param_file = ic_param_file
            
        if load_model_at_init:
            self.model.load_state_dict(torch.load(self.ic_param_file))
        elif save_model_at_init:
            torch.save(self.model.state_dict(), self.ic_param_file)
        
        self.rnn_mode = rnn_mode

        self.constancy_factor = constancy_factor
        self.tau_u = tau_u
        self.loud = loud
        self.model_type = model_type
        self.do_backprop = do_backprop
        self.k = np.array(k)
        self.k_ = torch.tensor(np.float32([self.k])).to(self.device)  
        self.ic = initial_state if initial_state is not None else 0.0  # Default to 0.0 if not provided
        self.fb_on_nan = fb_on_nan
        self.enable_combo = enable_combo
        self.auto_steps = auto_steps
        self.grad_less_steps = grad_less_steps
        self.runner_method_alias = runner_method_alias
        # Initialize low-pass filter
        self.u_lp = LPF(tau=tau_u)
        self.step_method_alias = step_method_alias

        if test_vec is None:
            self.test_vec = None
        elif isinstance(test_vec, np.ndarray):  
            self.test_vec = torch.tensor(np.float32(test_vec)).to(self.device)  
        elif isinstance(test_vec, torch.Tensor):  
            self.test_vec = test_vec.to(self.device)  
        else:
            raise TypeError(f"test_vec must be either a NumPy array or a PyTorch tensor, but got {type(test_vec)}")
        
        # Initialize records and states
        self.learning_rate = learning_rate

        # Reset the model and low-pass filter
        self.reset(silent=True)
    
    def reset(self, silent=False):
        if self.loud and not silent:
            print('model reset')

        self.model.reset_state()
        if self.model_type == 'torch':
            self.optimizer = self.optimizer_class(self.model.parameters(), 
                            lr=self.learning_rate, **self.optimizer_opts)
            self.model.to(self.device)
            if self.ic_param_file is not None:
                self.model.load_state_dict(torch.load(self.ic_param_file))

        self.records = SimpleNamespace(u=[], u_lp=[], test_output=[], extra_results=[])
        self.block_training_next_step = False
        self.u_lp.reset()

    def parse_criterion(self):
        if isinstance(self.criterion,str):
            if self.criterion == 'L1':
                self.criterion = nn.L1Loss()
            elif self.criterion == 'MSE':
                self.criterion = nn.MSELoss()
            else:
                raise ValueError
                
    def test_vec_eval(self):
        '''
        this method shuld cover any advanced evaluations that 
        are not related to the core dataset
        '''
        if self.test_vec is not None:
            self.model.eval()
            with torch.no_grad():
                test_output = self.model(self.k_*self.test_vec)
                self.records.test_output.append(test_output.cpu().detach().numpy())
    
    def opt_(self, u_t, y_t, constancy_factor=None, u_tm1=None):
            
            if constancy_factor is not None:
                y_t = (1-constancy_factor)*y_t + constancy_factor*u_tm1

            y_t_ = torch.tensor([float(y_t)], requires_grad=False).to(self.device)
            loss = self.criterion(u_t, y_t_)
            loss.backward()
            self.optimizer.step()

    def take_measurements(self, extra_measurements):
        '''
        this method should cover any advanced evaluations that 
        can be a function of the models partameters, its recurrent state or the hidden representation

        extra_measurements: a list of functions that take the model as an argument
        todo: add the recurrent state and the hidden representation as arguments
        '''
        results = []
        if extra_measurements is not None:
            for f in extra_measurements:
                results.append(f(self.model))
        
        return results

    def step(self, *args, **kwargs):
        '''
        this method should be used for the stepwise evaluation of the model
        it is a selector for step method
        '''
        if self.step_method_alias == 'step_vanilla':
            return self.step_vanilla(*args, **kwargs)
        elif self.step_method_alias == 'step_2stage':
            return self.step_step_by_step(*args, **kwargs)
        else:
            raise ValueError(f'step_method_alias: {self.step_method_alias} not recognized')
    
            
    def step_vanilla(self, y_t, state, extra_measurements=None, record=True):
        '''
        single step of continual learning 
        with optional training
        '''

        #unpacking the state
        x_tm1 = state[0]
        rnn_state = state[1] if self.rnn_mode else None

        model_input = self.k*x_tm1

        #hook for taking the first element of x_tm1 as u_tm1
        if self.constancy_factor is not None:
            u_tm1 = x_tm1[0]
            torch_u_tm1 = torch.tensor([float(u_tm1)], requires_grad=False).to(self.device)

        # applying model to the previous state; previous state relies on the previous sensory feedback (AKA y_{t-1})
        if self.model_type=='torch':
            self.model.train()
            self.optimizer.zero_grad() 
            model_input_ = torch.tensor(np.float32(model_input), requires_grad=False).to(self.device)

            #todo:d decide if the rnn_state is pased externally or not
            # model_input_ = (model_input_,) if not self.rnn_mode else (model_input_, rnn_state)
            model_input_ = (model_input_,) #if not self.rnn_mode else (model_input_, rnn_state)
            
            model_output = self.model(*model_input_)

            torch_u_t = model_output if not self.rnn_mode else model_output[0]
            rnn_state = model_output[1] if self.rnn_mode else None

            u_t = torch_u_t.cpu().detach().numpy().squeeze()
        elif self.model_type=='numpy':
            u_t = self.model(model_input)
        else:
            raise ValueError(f'model_type: {self.model_type} not recognized')
        
            
        self.u_lp.step(u_t, silent=True)

        #updating error if applicable (that is if y_t is not nan)
        #then:
        #taking training step on model parameters based on the current sensory feedback:
        # target is y_t while and the  prediction which is based on the previous state: 
        cond = not np.isnan(y_t)
        if cond:
            err_t = (y_t - self.u_lp.state)
            if self.do_backprop and not self.block_training_next_step:
                self.opt_(torch_u_t,y_t,
                          **({'constancy_factor': self.constancy_factor, 'u_tm1':torch_u_tm1} if self.constancy_factor is not None else {}))
        elif self.constancy_factor is not None:
            y_t = u_tm1
            err_t = np.zeros_like(self.u_lp.state)
            if self.do_backprop and not self.block_training_next_step:
                self.opt_(torch_u_t,y_t,
                          **({'constancy_factor': self.constancy_factor, 'u_tm1':torch_u_tm1} if self.constancy_factor is not None else {}))                 
        else:
            err_t = np.zeros_like(self.u_lp.state)

        self.block_training_next_step = not cond #todo: currently overriden at the top level. fix this (either remove here or check why needed)
        y_t = y_t if not np.isnan(y_t) else self.fb_on_nan(self.u_lp.state,err_t)

        #preparing the state for the next step    
        x_t = np.array([self.u_lp.state,
                        y_t,
                        err_t]+([self.u_lp.state + err_t] if self.enable_combo else []))

        if record:
            self.records.u.append(u_t)
            self.records.u_lp.append(self.u_lp.state)

            self.records.extra_results.append(
                self.take_measurements(extra_measurements))

        return (x_t, rnn_state) if self.rnn_mode else (x_t,)
    
    def step_2stage(self, y_t, state, extra_measurements=None, record=True):
        '''
        supported for torch models only
        '''
        if self.model_type != 'torch':
            raise ValueError('step_2stage is supported for torch models only')

        y_tm1, c_tm1 = state

        #posterior probabilities prediction, w/o observing y_t:
        r_tm1 = r_model(y_tm1) if y_tm1 is not None else None
        c_t = c_model(r_tm1 if r_tm1 is not None else c_tm1)
        u_t = x_model(c_t)
        #autoencoder training step (responsibilities)
        r_t = r_model(y_t)
        y_t_hat = x_model(r_t)
        loss_ae = criterion_ae(y_t_hat, y_t)
        loss_ae.backward()
        optimizer_ae.step() #training x_model and r_model
        #filtering step (b): update; todo, where we locate this step, here or before the AE step
        r_tm1_ = r_model(y_tm1)
        c_t_ = c_model(r_tm1_)
        u_t_ = x_model(c_t_)
        loss_pred = criterion_pred(u_t_, y_t)
        loss_pred.backward()
        optimizer_pred.step() #training c_model

        #taking records #todo: refactor to avoid code duplication
        if record:
            self.records.u.append(u_t)
            self.records.u_lp.append(self.u_lp.state)

        #returning the state
        return (y_t, c_t)

    def run(self, scenario, **kwargs):

        # scenario = kwargs.pop('scenario', None)# todo: doublecheck interface
        if scenario is None:
            raise ValueError('scenario must be provided')
        if type(scenario) == list:
                y = parse_samples(scenario)
        else:
            raise NotImplementedError #in future we will also support parsed lists
        
        kwargs['y'] = y

        if self.runner_method_alias == 'step_by_step':
            return self.run_step_by_step(**kwargs)
        elif self.runner_method_alias == 'blackbox':
            return self.run_black_box(**kwargs)
        else:
            raise ValueError(f'runner_method_alias: {self.runner_method_alias} not recognized')
        
    def run_black_box(self, y=None,
            do_return=True,
            test_vec=None,
            extra_measurements=None):
        
        output = self.model(y) 
        return output.records, self.model

    def run_step_by_step(self,y=None,
            do_return=True,
            test_vec=None,
            extra_measurements=None):

        this_state = self.initial_state
        for t, y_t in enumerate(y):
            self.test_vec_eval()
            this_state = self.step(y_t,
                                   this_state,
                                   extra_measurements=extra_measurements,
                                   record=True) 
            for _ in range(self.auto_steps):
                this_state = self.step(np.nan,
                            this_state,
                            extra_measurements=extra_measurements,
                            record=False)
            for _ in range(self.grad_less_steps):
                self.block_training_next_step = True
                this_state = self.step(y_t,
                            this_state,
                            extra_measurements=extra_measurements,
                            record=True)
            self.block_training_next_step = False #todo - fix this           
            
        if do_return:
            return copy(self.records), self.model
        
    def run_multiple(self,playlist,             
            test_vec=None,
            extra_measurements=None,
            silent=False):
        results = {}
        
        for name, scenario in playlist.items():
            self.reset(silent=True) #verbosing done below anyway, therefore 'silent' here
            if not silent:
                print(f'running scenario: {name}')

            results[name], _model = self.run(scenario,test_vec=test_vec,
                                             extra_measurements=extra_measurements)
        return results
    

def wrap_runner_for_optimization(model_class=None,
                                 fixed_params={},
                                 optim_params_mapping=[],
                                 custom_param_mappings=[], 
                                 postprocessing_fun=None, 
                                 runner_class=Runner):
    
    '''
    wrapper for the runner construction and application
    for the purpose of optimization

    model_class: class
        the class of the core model to be optimized
    fixed_params: dict
        parameters that are fixed and not subject to optimization
    optim_params_mapping: list of tuples
        a list of tuples (param_cathegory,param_name) or (param_cathegory,param_name,preprocessing_function)
    custom_param_mappings: a more general interface for parameter mapping: list of dicts with  keys 'fun','cathergory','param_name':
        each fun must be a function of all the optimization parameters and return a value that will be subsequently passed to one of the optim_params categories 
    '''
    
    known_param_categories = ['model','runner','postprocessing','custom']

    for foo in optim_params_mapping:
        param_cathegory = foo[0]
        if param_cathegory not in known_param_categories:
            raise ValueError(f'param_cathegory: {param_cathegory} not recognized')
    
    model_args = fixed_params['model'] if 'model' in fixed_params else {}
    runner_args = fixed_params['runner'] if 'runner' in fixed_params else {}
    postprocessing_args = fixed_params['postprocessing'] if 'postprocessing' in fixed_params else {}

    def wrapped_runner(stimulus,param_vals):
        
        optim_params = {param_cathegory: {} for param_cathegory in known_param_categories}
        
        #basic assingment of param_vals to the optim_params
        for i, param_val in enumerate(param_vals):
            if optim_params_mapping[i][0] == 'custom':
                continue

            if len(optim_params_mapping[i]) == 2:
                param_cathegory, param_name = optim_params_mapping[i]
                optim_params[param_cathegory][param_name] = param_val
            elif len(optim_params_mapping[i]) == 3:   
                param_cathegory, param_name, preprocessing_function = optim_params_mapping[i]
                optim_params[param_cathegory][param_name] = preprocessing_function(param_val)

        #performing custom mappings that might involve multiple input parameters being transformed into a single output parameter and vice versa
        param_vals_dict = {optim_params_mapping_el[1]: param_val for optim_params_mapping_el, param_val in zip(optim_params_mapping,param_vals)}   

        for custom_param_mappting in custom_param_mappings:
            optim_params[custom_param_mappting['cathegory']][custom_param_mappting['param_name']] = custom_param_mappting['fun'](param_vals_dict)

        runner = runner_class(model_class=model_class, model_construct_args={**model_args, **optim_params['model']},
                        test_vec=None,
                        **{**runner_args,**optim_params['runner']})
        
        result = runner.run(stimulus)[0].u_lp

        if postprocessing_fun is not None:
            result = postprocessing_fun(result,**{**postprocessing_args, **optim_params['postprocessing']})

        return result
    
    return wrapped_runner
