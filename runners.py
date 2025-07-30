import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import copy
from types import SimpleNamespace
import ntk_utils

from models import MLP
from dsp_utils import LPF, parse_samples


class Runner:
    def __init__(self,
                model=None,
                model_class=MLP,
                model_construct_args=None,
                models=None,
                rnn_mode = False,
                optimizers=None,
                optimizer_class=None,
                optimizer_opts={},
                learning_rate = None, #1e-5,
                criterion='MSE',
                criteria = None,
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
                noise_spec = {},
                filter_spec = {},
                scaling_spec = {},
                test_vec=None,
                initial_state=None,
                apply_initial_state=True,
                load_model_at_init=False,
                save_model_at_init=True,
                fb_on_nan=lambda x,y:0.,
                auto_steps=0,
                grad_less_steps=0,
                aux_parallel_model=None, #not in use
                info=None, #not in use
                sigma_noi = 0.0, #not in use
                take_lin_measurements=False,
                lin_measurement_opts={},
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
            noise_spec (dict): the following keys are supported:
                'noi_x' (float): Noise to be added to the model input (x).
                'noi_u' (float): Noise to be added to the model output (u).
                'noi_y' (float): Noise to be added to the target output (y).
                'noi_post_u' (float): Noise to be added to the output (u) after closing the loop. this component does not go into the feedback loop.
            test_vec (array): Test vector for evaluation.
            initial_state (array): Initial state of the system.
            apply_initial_state (bool): If True, applies the initial state.
            load_model_at_init (bool): If True, loads the model parameters from the file at initialization.
            save_model_at_init (bool): If True, saves the model parameters to the file at initialization.
            fb_on_nan (function): Function to handle NaN values in the feedback.
            auto_steps (int): Number of underhood automatic steps to take after each recorded step.
            grad_less_steps (int): Number of steps to take without gradient updates after each recorded step.
            aux_parallel_model (TBD): auxilary model connected in parallel to the main model
            info (dict): Additional information (not used); for interface consistency.
        """
        if sigma_noi is not None and sigma_noi > 1e-100:
            raise ValueError('sigma_noi is not supported anymore, use noise_spec instead')

        self.models = models
        self.optimizers = optimizers
        self.criteria = criteria

        if optimizers is None:
            self.optimizer_class = optimizer_class if optimizer_class is not None else optim.SGD
            self.optimizer_opts = optimizer_opts
            
        self.criterion = criterion

        if criteria is None:
            if model_type == 'torch':
                self.parse_criterion()
        if models is None:
            self.model = model_class(**model_construct_args) if model is None else model            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device    

        if apply_initial_state:   
            self.initial_state = np.zeros(model_construct_args['n_inputs']) if initial_state is None else initial_state
        else:
            self.initial_state = None
            
        self.ic_param_file = ic_param_file
            
        # if load_model_at_init:
        #     self.model.load_state_dict(torch.load(self.ic_param_file))
        # elif save_model_at_init:
        #     torch.save(self.model.state_dict(), self.ic_param_file)

        if load_model_at_init:
            # if user supplied a single model, load it…
            if self.models is None:
                self.model.load_state_dict(torch.load(self.ic_param_file))
           # …otherwise load each sub‐model under a separate file suffix
            else:
                for name, m in vars(self.models).items():
                    path = f"{self.ic_param_file}_{name}.pth"
                    m.load_state_dict(torch.load(path))
        elif save_model_at_init:
            # save the standalone model…
            if self.models is None:
                torch.save(self.model.state_dict(), self.ic_param_file)
            # …or save each sub‐model to its own file
            else:
                for name, m in vars(self.models).items():
                    path = f"{self.ic_param_file}_{name}.pth"
                    torch.save(m.state_dict(), path)
        

        self.rnn_mode = rnn_mode
        self.optimizers = optimizers #TODO: add enhance support for multiple optimizers
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
        self.aux_parallel_model = aux_parallel_model 

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

        self.noise_spec = noise_spec
        self.filter_spec = filter_spec
        self.scaling_spec = scaling_spec

        self.filters = {}
        for key in self.filter_spec:
            self.filters[key] = LPF(tau=self.filter_spec[key])
        # Reset the model and low-pass filter
        self.take_lin_measurements = take_lin_measurements
        self.lin_measurement_opts = lin_measurement_opts
        self.reset(silent=True)
    
    def reset(self, silent=False):
        if self.loud and not silent:
            print('model reset')

        if self.models is None:
            self.model.reset_state()
        else:
            for name, m in vars(self.models).items():
                m.reset_state()

        if self.model_type == 'torch':
            if self.optimizers is None: #checking that there is no provided optimizers
                if 'parameter_groups_opts' in self.optimizer_opts:
                    # only supported for model's subblocks being called models
                    param_groups = []
                    for idx, sub_model in enumerate(self.model.models):
                        param_groups.append({
                            "params": sub_model.parameters(),
                            **self.optimizer_opts['parameter_groups_opts'][idx]
                        })
                    self.optimizer = self.optimizer_class(param_groups)
                    if self.learning_rate is not None:
                        raise ValueError('stand alone learning_rate is not supported when parameter_groups_opts is used')
                else:
                    self.optimizer = self.optimizer_class(self.model.parameters(), 
                                lr=self.learning_rate, **self.optimizer_opts)
                
            if self.models is None: #checking that there is no provided models
                self.model.to(self.device)
                if self.ic_param_file is not None:
                    self.model.load_state_dict(torch.load(self.ic_param_file))
            else:
                for name, m in vars(self.models).items():
                    m.to(self.device)
                    path = f"{self.ic_param_file}_{name}.pth"
                    m.load_state_dict(torch.load(path))

        self.records = SimpleNamespace(u=[], u_lp=[], test_output=[], extra_results=[])
        self.block_training_next_step = False
        self.u_lp.reset()
        if self.aux_parallel_model is not None:
            self.aux_parallel_model.reset_state()
        for key in self.filters:
            self.filters[key].reset()
        if self.take_lin_measurements:
            self._stepwise_recorder = []  # For storing stepwise records if needed
            x_grid = self.lin_measurement_opts.get('x_grid', None)
            K = ntk_utils.compute_ntk_matrix_np(self.model, x_grid, device=self.device)
            H = ntk_utils.compute_dntk_dxj_tensor_np(self.model, x_grid, device=self.device)
            self._initial_recordings = {'K': K, 'H': H}

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
    
    def opt_(self, u_t, y_t, constancy_factor=None, u_tm1=None,weight_decay_only=False):
            if self.model_type=='torch':
                if weight_decay_only and hasattr(self, 'optimizer'):
                    self.optimizer.zero_grad()
                    self.optimizer.step()
                    return
                
                if constancy_factor is not None:
                    y_t = (1-constancy_factor)*y_t + constancy_factor*u_tm1

                y_t_ = torch.tensor([float(y_t)], requires_grad=False).to(self.device)
                loss = self.criterion(u_t, y_t_)
                loss.backward()

                #TODO: ensure optimiser is packed into a list in all cases and remove this check
                if self.optimizers is not None: 
                    for optimizer in self.optimizers:
                        optimizer.step()
                else:
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
            return self.step_2stage(*args, **kwargs)
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

        if self.take_lin_measurements:
            #TODO: this will only work correctly for k = [0,0,0,1]; shold be generalized
            Jx_tm1 = ntk_utils.compute_dfdx_tensor_np(self.model, model_input, device=self.device)

        if 'noi_x' in self.noise_spec:
            model_input += self.noise_spec['noi_x'] * np.random.randn(*model_input.shape)

        if 'tau_x' in self.filter_spec:
            model_input = self.filters['tau_x'].step(model_input, silent=False)
                
        #hook for taking the first element of x_tm1 as u_tm1
        if self.constancy_factor is not None:
            u_tm1 = x_tm1[0]
            torch_u_tm1 = torch.tensor([float(u_tm1)], requires_grad=False).to(self.device)

        # applying model to the previous state; previous state relies on the previous sensory feedback (AKA y_{t-1})
        if self.model_type=='torch':
            self.model.train()

            if self.optimizers is not None:
                for optimizer in self.optimizers:
                    optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()
                
            model_input_ = torch.tensor(np.float32(model_input), requires_grad=False).to(self.device)

            #TODO:d decide if the rnn_state is pased externally or not
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
        
        if self.aux_parallel_model is not None:
            aux_output = self.aux_parallel_model.current_state()
            u_t += aux_output


        if 'noi_u' in self.noise_spec:
            u_t += self.noise_spec['noi_u'] * np.random.randn(*u_t.shape)
            
        self.u_lp.step(u_t, silent=True)

        #updating error if applicable (that is if y_t is not nan)
        #then:
        #taking training step on model parameters based on the current sensory feedback:
        # target is y_t while and the  prediction which is based on the previous state: 
        cond = not np.isnan(y_t)
        if cond:
            if 'noi_y' in self.noise_spec:
                y_t += self.noise_spec['noi_y'] * np.random.randn(*y_t.shape)

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
            self.opt_(None,None,weight_decay_only=True)
            err_t = np.zeros_like(self.u_lp.state)

        if self.aux_parallel_model is not None:
            _ = self.aux_parallel_model.step(err_t)

        self.block_training_next_step = not cond #todo: currently overriden at the top level. fix this (either remove here or check why needed)
        y_t = y_t if not np.isnan(y_t) else self.fb_on_nan(self.u_lp.state,err_t)

        #preparing the state for the next step 
        u_fb = self.u_lp.state #TODO: consider removing u_lp and working with u_t
        if 'tau_u_fb' in self.filter_spec:
            u_fb = self.filters['tau_u_fb'].step(u_fb, silent=False)
        if 'scaling_u_fb' in self.scaling_spec:
            u_fb *= self.scaling_spec['scaling_u_fb']
            
        x_t = np.array([u_fb,
                        y_t,
                        err_t]+([u_fb + err_t] if self.enable_combo else []))

        if 'noi_post_u' in self.noise_spec:
            n_post_u = self.noise_spec['noi_post_u'] * np.random.randn(*u_t.shape)
        else:
            n_post_u = 0.0

        if record:
            self.records.u.append(u_t+n_post_u)
            self.records.u_lp.append(self.u_lp.state+n_post_u)

            self.records.extra_results.append(
                self.take_measurements(extra_measurements))
        if self.take_lin_measurements:
            #take optimizer related measurements if available
            #if optimizer has a method get_global_lr
            if hasattr(self.optimizers, 'get_global_lr'):
                this_lr = self.optimizers.get_global_lr()
            else:
                this_lr = None

            #TODO: verify the timing
            self._stepwise_recorder.append({'e': err_t, 'u': u_t, 'y': y_t, 'x': x_tm1, 'Jx': Jx_tm1, 'lr': this_lr,})

        return (x_t, rnn_state) if self.rnn_mode else (x_t,)
    
    def step_2stage(self, y_t, state, extra_measurements=None, record=True):
        '''
        Two-stage continual learning step for rcx models.

        Args:
            y_t: current observation (float) or None if missing
            state: tuple (y_{t-1}, c_{t-1})
            extra_measurements: list of callables for diagnostics
            record: whether to record outputs

        Returns:
            new state (y_t, c_t)
        '''
        if self.model_type != 'torch':
            raise ValueError('step_2stage is supported for torch models only')

        # Unpack previous state
        y_tm1, c_tm1 = state
        # print('debug:   y_tm1:    ',y_tm1)
        # if np.isnan(y_tm1):
        #     y_tm1 = None
        # if np.isnan(y_t):
        #     y_t = None

        # ctm1 to tensor
        if not isinstance(c_tm1, torch.Tensor):
            c_tm1_tensor = torch.tensor(c_tm1) #TODO: doublecheck for redundancy
        else:
            c_tm1_tensor = c_tm1
        c_tm1_tensor = c_tm1_tensor.to(self.device)

        c_tm1 = c_tm1_tensor

        # Aliases
        r_model = self.models.r_model
        c_model = self.models.c_model
        x_model = self.models.x_model
        optim_ae = self.optimizers.optimizer_ae
        optim_pred = self.optimizers.optimizer_pred
        crit_ae = self.criteria.ae
        crit_pred = self.criteria.pred

        # Stage 1: Predict responsibilities, context, and output
        if not np.isnan(y_tm1):
            y_tm1_tensor = torch.tensor([[float(y_tm1)]], device=self.device)
            r_tm1 = r_model(y_tm1_tensor)
        else:
            r_tm1 = None
        c_t = c_model(r_tm1 if r_tm1 is not None else c_tm1)
        u_t_tensor = x_model(c_t)
        u_t = u_t_tensor.cpu().detach().numpy().squeeze()

        # Update low-pass filter
        self.u_lp.step(u_t, silent=True)

        # Stage 2a: Autoencoder update (train r_model + x_model)
        if not np.isnan(y_t):
            y_t_tensor = torch.tensor([[float(y_t)]], device=self.device)
            optim_ae.zero_grad()
            r_t = r_model(y_t_tensor)
            y_hat = x_model(r_t)
            loss_ae = crit_ae(y_hat, y_t_tensor)
            loss_ae.backward()
            optim_ae.step()

        # Stage 2b: Predictor update (train c_model)
        if not np.isnan(y_t) and not np.isnan(y_tm1):
            y_tm1_tensor = torch.tensor([[float(y_tm1)]], device=self.device)
            optim_pred.zero_grad()
            r_tm1_pred = r_model(y_tm1_tensor)
            c_pred = c_model(r_tm1_pred)
            u_pred = x_model(c_pred)
            loss_pred = crit_pred(u_pred, y_t_tensor)
            loss_pred.backward()
            optim_pred.step()

        # Record outputs and diagnostics
        if record:
            self.records.u.append(u_t)
            self.records.u_lp.append(self.u_lp.state)
            self.records.extra_results.append(self.take_measurements(extra_measurements))

        return (y_t, c_t)

    
    # def step_2stage(self, y_t, state, extra_measurements=None, record=True):
    #     '''
    #     supported for torch models only
    #     '''
    #     if self.model_type != 'torch':
    #         raise ValueError('step_2stage is supported for torch models only')

    #     x_tm1, c_tm1 = state

    #     if self.k is not None:
    #         raise NotImplementedError('k is not supported in step_2stage')
    #     else:
    #         y_tm1 = x_tm1

    #     #aliases
    #     r_model, c_model, x_model = self.models.r_model, self.models.c_model, self.models.x_model
    #     optimizer_ae, optimizer_pred = self.optimizers.optimizer_ae, self.optimizers.optimizer_pred
    #     criterion_ae, criterion_pred = self.criteria.ae, self.criteria.pred

    #     #posterior probabilities prediction, w/o observing y_t:
    #     r_tm1 = r_model(y_tm1) if y_tm1 is not None else None #y_tm1 is None in channel trials
    #     c_t = c_model(r_tm1 if r_tm1 is not None else c_tm1)
    #     u_t = x_model(c_t)

    #     #autoencoder training step (responsibilities)
    #     if y_t is not None:
    #         r_t = r_model(y_t)
    #         y_t_hat = x_model(r_t)

    #         loss_ae = criterion_ae(y_t_hat, y_t)
    #         loss_ae.backward()
    #         optimizer_ae.step() #training x_model and r_model

    #     #filtering step (b): update; TODO, where we locate this step, here or before the AE step
    #     if y_t is not None and y_tm1 is not None:
    #         r_tm1_ = r_model(y_tm1)
    #         c_t_ = c_model(r_tm1_)
    #         u_t_ = x_model(c_t_)

    #         loss_pred = criterion_pred(u_t_, y_t)
    #         loss_pred.backward()
    #         optimizer_pred.step() #training c_model

    #     #taking records #todo: refactor to avoid code duplication
    #     if record:
    #         self.records.u.append(u_t)
    #         self.records.u_lp.append(self.u_lp.state)

    #     #returning the state
    #     return (y_t, c_t)

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
            return copy(self.records), (self.model if self.models is None else self.models)
        
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

    def wrapped_runner(stimulus,param_vals,seed=0,return_runner=False):
        
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
        if seed is not None:
            torch.manual_seed(seed)  
            np.random.seed(seed)
        result = runner.run(stimulus)[0].u_lp

        if postprocessing_fun is not None:
            result = postprocessing_fun(result,**{**postprocessing_args, **optim_params['postprocessing']})

        if return_runner:
            return result, runner
        else:
            return result
    
    return wrapped_runner
