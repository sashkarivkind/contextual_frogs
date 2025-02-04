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
                 optimizer=None,
                 learning_rate = 1e-5,
                 criterion='MSE',
                 device=None,
                 ic_param_file='model_parameters.pth',
                 create_ic_param_file=True,
                 tau_u=1.0,
                 loud=True,
                 model_type='torch',
                 do_backprop=True,
                 k=[0,1.,0],
                 enable_combo=False,
                 sigma_noi = 0,
                 test_vec=None,
                 initial_state=None,
                load_model_at_init=False,
                save_model_at_init=True,
                fb_on_nan=lambda x,y:0.,
                auto_steps=0,
                info=None, #not in use
                ):
        """
        Initialize the Runner class.

        Args:
            model: The machine learning model (PyTorch or NumPy-based).
            optimizer: Optimizer used for training the model (e.g., PyTorch optimizer).
            criterion: Loss function used for training.
            device (str): Device to use ('cpu' or 'cuda').
            ic_param_file (str): Path to the file with initial model parameters.
            tau_u (float): Time constant for the low-pass filter.
            loud (bool): If True, prints reset messages.
            model_type (str): Type of model, either 'torch' or 'numpy'.
            do_backprop (bool): Whether to perform backpropagation during training.
            k (float): Coefficient for model input scaling.
            initial_state (Optional): Initial state of the system (used in `run`).
        """
        if np.abs(sigma_noi)>1e-20:
            raise NotImplementedError
        self.optimizer = optimizer
        self.criterion = criterion
        if model_type == 'torch':
            self.parse_criterion()
        
        self.model = model_class(**model_construct_args) if model is None else model            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device       
        self.initial_state = np.zeros(model_construct_args['n_inputs']) if initial_state is None else initial_state
        
        self.ic_param_file = ic_param_file
            
        if load_model_at_init:
            self.model.load_state_dict(torch.load(self.ic_param_file))
        elif save_model_at_init:
            torch.save(self.model.state_dict(), self.ic_param_file)
        
        
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
        # Initialize low-pass filter
        self.u_lp = LPF(tau=tau_u)
        
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
        self.reset()
    
    def reset(self, silent=False):
        if self.loud and not silent:
            print('model reset')
        if self.model_type == 'torch':
            self.optimizer = optim.SGD(self.model.parameters(), 
                            lr=self.learning_rate)
            self.model.to(self.device)
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
    
    def opt_(self, u_t, y_t):
            y_t = torch.tensor([float(y_t)], requires_grad=False).to(self.device)
            loss = self.criterion(u_t, y_t)
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

            
    def step(self, y_t, x_tm1, extra_measurements=None, record=True):
        '''
        single step of continual learning 
        with optional training
        '''

        model_input = self.k*x_tm1
        
        if self.model_type=='torch':
            self.model.train()
            self.optimizer.zero_grad()        
            torch_u_t = self.model(
                torch.tensor(np.float32(model_input), requires_grad=False).to(self.device)
            )
            u_t = torch_u_t.cpu().detach().numpy().squeeze()
        elif self.model_type=='numpy':
            u_t = model(model_input)
        else:
            raise ValueError
            
        self.u_lp.step(u_t, silent=True)

        cond = not np.isnan(y_t)
        if cond:
            err_t = (y_t - self.u_lp.state)
            if self.do_backprop and not self.block_training_next_step:
                self.opt_(torch_u_t,y_t)
                        
        else:
            err_t = np.zeros_like(self.u_lp.state)

        self.block_training_next_step = not cond
        y_t = y_t if not np.isnan(y_t) else self.fb_on_nan(self.u_lp.state,err_t)
            
        x_t = np.array([self.u_lp.state,
                        y_t,
                        err_t]+([self.u_lp.state + err_t] if self.enable_combo else []))

        if record:
            self.records.u.append(u_t)
            self.records.u_lp.append(self.u_lp.state)

            self.records.extra_results.append(
                self.take_measurements(extra_measurements))

        return x_t 
    
    def run(self,y,
            do_return=True,
            test_vec=None,
            extra_measurements=None):
        '''
        full training session
        test_vec: optional test vector
        measurements: optional measurements
        '''
        this_state = self.initial_state
        for t, y_t in enumerate(y):
            self.test_vec_eval()
            this_state = self.step(y_t,
                                   this_state,
                                   extra_measurements=extra_measurements,
                                   record=True) 
            for _ in range(self.auto_steps):
                self.step(np.nan,
                          this_state,
                          extra_measurements=extra_measurements,
                          record=False)
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
            if type(scenario) == list:
                to_play = parse_samples(scenario)
            else:
                raise NotImplementedError #in future we will also support parsed lists
                
            results[name], _model = self.run(to_play,test_vec=test_vec,extra_measurements=extra_measurements)
        return results