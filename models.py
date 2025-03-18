import torch
import torch.nn as nn
import numpy as np

def nl_selector(nl):
    if isinstance(nl,str):
        if nl == 'relu':
            nl = nn.ReLU
        elif nl == 'symrelu':
            nl = SymReLU
        elif nl == 'tanh':
            nl = nn.Tanh
        elif nl == 'sigmoid':
            nl = nn.Sigmoid
        elif nl is None:
            nl = lambda x: x
        else:
            raise ValueError('unkonow nonlinearity')
    return nl

class ModelForRunner():
    #abstract class with reset_state method
    def reset_state(self):
        raise NotImplementedError

class PlainRNN(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, g=1.0, nl='relu', trainable=False):
        super(PlainRNN, self).__init__()
        self.nl = nl_selector(nl)()
        self.hidden_size = hidden_size
        
        # Initialize fixed weights (non-trainable)
        self.W = nn.Parameter(torch.normal(mean=0, std=g/np.sqrt(hidden_size), size=(hidden_size, hidden_size)), requires_grad=trainable)
        self.U = nn.Parameter(torch.normal(mean=0, std=1., size=(input_size, hidden_size)), requires_grad=trainable)
    
    def forward(self, x, h):
        """
        x: Input tensor of shape (batch_size, input_size)
        h: Hidden state tensor of shape (batch_size, hidden_size)
        """
        h = self.nl(
            torch.matmul(h, self.W) + torch.matmul(x, self.U))
                              
        return h

class ParallelMLP(nn.Module, ModelForRunner):
    # parallel MLPs receive list of parameter dicts, each is used to initialize a separate MLP
    # raises error in n_outs is different for different MLPs
    # or if n_inputs is different for different MLPs when shared_input=True
    # if shared_input is false then the input is sliced according to n_inputs in each parameter set, and each MLP receives a different slice 
    # the outputs are summed
    def __init__(self, parameter_sets=None, n_inputs=None, shared_input=True, info=None):
        super(ParallelMLP, self).__init__()
        self.models = nn.ModuleList([MLP(**params) for params in parameter_sets])
        self.n_outs = self.models[0].output_layer.out_features
        self.shared_input = shared_input
        for model in self.models:
            if model.output_layer.out_features != self.n_outs:
                raise ValueError('inconsistent n_outs in parallel MLPs')
        
        if shared_input:
            self.n_inputs = self.models[0].input_layer.in_features
            for model in self.models:
                if model.input_layer.in_features != self.n_inputs:
                    raise ValueError('inconsistent n_inputs in parallel MLPs')
        else:
            self.n_inputs = [model.input_layer.in_features for model in self.models]
        
    def forward(self, x):
        if self.shared_input:
            return torch.stack([model(x) for model in self.models], dim=0).sum(dim=0)
        else:
            #prepare input slices for each model
            counter = 0
            slices = []
            for n_inputs in self.n_inputs:
                slices.append(x[:, counter:counter+n_inputs])
                counter += n_inputs
            return torch.stack([model(slice) for model, slice in zip(self.models, slices)], dim=0).sum(dim=0)

    def reset_state(self):
        for model in self.models:
            model.reset_state()

class MLP(nn.Module, ModelForRunner):
    def __init__(self, n_inputs=None, n_hidden=None, n_outs=None, n_layers=1, nl='tanh', en_bias=True, prescaling=None, main_gain=None,
                 b_low=None, b_high=None, first_layer_init='default', skip_gain=None, first_layer_weights_trainable=False, out_layer_init='default',
                 info=None):
        super(MLP, self).__init__()
        

        if prescaling is not None:
            #ensure prescaling is same size as n_inputs
            if len(prescaling) != n_inputs:
                raise ValueError('prescaling must be same size as n_inputs')
            prescaling = torch.tensor(prescaling, dtype=torch.float32, requires_grad=False)
        else:
            prescaling = torch.ones(n_inputs, dtype=torch.float32, requires_grad=False)
        #prescaling is not a layer and therefre should be registered as buffer
        self.register_buffer('prescaling', prescaling)

        if main_gain  is not None:
            #ensure outscaling is same size as n_outs
            if n_outs == 1 and not isinstance(main_gain, list) and not isinstance(main_gain, np.ndarray):
                main_gain = [main_gain]
            if len(main_gain) != n_outs:
                raise ValueError('outscaling must be same size as n_outs')
            main_gain  = torch.tensor(main_gain, dtype=torch.float32, requires_grad=False)
        else:
            main_gain  = torch.ones(n_outs, dtype=torch.float32, requires_grad=False)

        skip_gain = skip_gain if skip_gain is not None else 0  # Default gain is 0

        if isinstance(skip_gain, list) or isinstance(skip_gain, np.ndarray):
            skip_gain = torch.tensor(skip_gain, dtype=torch.float32, requires_grad=False)
        else:
            skip_gain = torch.tensor([skip_gain]*n_outs, dtype=torch.float32, requires_grad=False)

        #prescaling, main_gain, skip_gain are not layers and therefre should be registered as buffer
        self.register_buffer('prescaling', prescaling)
        self.register_buffer('main_gain', main_gain)
        self.register_buffer('skip_gain', skip_gain)

        nl = nl_selector(nl)
        self.activation = nl()

        # Input layer
        if b_low is None and b_high is not None:
            b_low = -b_high

        custom_first_bias = b_low is not None and b_high is not None

        self.n_layers = n_layers

        self.input_layer = nn.Linear(n_inputs, n_hidden, bias=en_bias or custom_first_bias)
        
        if first_layer_init == 'ones':
            nn.init.constant_(self.input_layer.weight, 1.0)          
        elif first_layer_init == 'uniform_unity':
            nn.init.uniform_(self.input_layer.weight, -1, 1)
        elif first_layer_init == 'default':
            pass
        else:
            raise ValueError('Unknown first_layer_init')

        self.input_layer.weight.requires_grad = first_layer_weights_trainable

        if custom_first_bias:
            bias_values = torch.linspace(b_low, b_high, n_hidden)
            with torch.no_grad():
                self.input_layer.bias.copy_(bias_values)
            self.input_layer.bias.requires_grad = False

        self.hidden_layers = nn.ModuleList([
            nn.Linear(n_hidden, n_hidden, bias=en_bias) for _ in range(n_layers - 1)
        ])

        self.output_layer = nn.Linear(n_hidden, n_outs, bias=en_bias)

        if out_layer_init == 'ones':
            nn.init.constant_(self.output_layer.weight, 1.0)
        elif out_layer_init == 'uniform_unity':
            nn.init.uniform_(self.output_layer.weight, -1, 1)
        elif out_layer_init == 'zeros':
            nn.init.constant_(self.output_layer.weight, 0.0)
        elif out_layer_init == 'default':
            pass

    def forward(self, x):
        
        if self.prescaling is not None:
            x = x * self.prescaling

        x_res = torch.sum(self.skip_gain *x, dim=-1, keepdim=True)

        x = self.input_layer(x)
        x = self.activation(x)

        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
        
        x = self.main_gain  * self.output_layer(x) +  x_res  
        return x
    
    def reset_state(self):
        pass
    
    


class SymReLU(torch.nn.Module):
    def __init__(self, threshold=1):
        super(SymReLU, self).__init__()
        self.threshold = threshold
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.threshold - torch.abs(x))


class OneOverSqr(torch.nn.Module):
    def __init__(self, w=1, c=0, normalize=False):
        super(OneOverSqr, self).__init__()
        self.w = w
        self.c = c
        self.normalize = normalize
        self.norm_factor = 1.0 / (2 * w) if normalize else 1.0

    def forward(self, x):
        return self.norm_factor*(1 + (x / self.w).pow(2)).reciprocal() + self.c


class GaussianKernel(torch.nn.Module):
    def __init__(self, w=1, c=0, normalize=False):
        super(GaussianKernel, self).__init__()
        self.w = w
        self.c = c
        self.normalize = normalize
        self.norm_factor = 1.0 / (2 * w) if normalize else 1.0

    def forward(self, x):
        return self.norm_factor*torch.exp(-0.5 * (x / self.w).pow(2)) + self.c
    
class RectangularKernel(torch.nn.Module):
    def __init__(self, w=1, c=0, normalize=False):
        super(RectangularKernel, self).__init__()
        self.w = w
        self.c = c
        self.normalize = normalize
        self.norm_factor = 1.0 / (2 * w) if normalize else 1.0

    def forward(self, x):
        return self.norm_factor*(torch.abs(x) <= self.w).float() + self.c


class DualRateModel(ModelForRunner):
    def __init__(self, a_s: float, b_s: float, a_f: float, b_f: float, input_proj_vec=None, info=None):
        """
        Initialize the dual-rate model parameters.
        :param a_s: Retention factor for the slow process (close to 1)
        :param b_s: Learning rate for the slow process (small value)
        :param a_f: Retention factor for the fast process (close to 0)
        :param b_f: Learning rate for the fast process (larger than b_s)
        """
        self.a_s = a_s
        self.b_s = b_s
        self.a_f = a_f
        self.b_f = b_f
        self.input_proj_vec = np.array(input_proj_vec) if input_proj_vec is not None else None
        self.reset_state()

    def reset_state(self):
        """Reset the model's adaptation states."""
        self.x_s = 0.0  # Slow process state
        self.x_f = 0.0  # Fast process state

    def step(self, e: float) -> float:
        """
        Update the model's states given an error signal e.
        :param e: The error signal at the current step.
        :return: The total adaptation response.
        """
        if self.input_proj_vec is not None:
            e = np.dot(self.input_proj_vec, e)
        self.x_s = self.a_s * self.x_s + self.b_s * e
        self.x_f = self.a_f * self.x_f + self.b_f * e
        return self.x_s + self.x_f

    def __call__(self, e: float) -> float:
        """
        Allows the model to be called as model(e), equivalent to model.step(e).
        :param e: The error signal.
        :return: The total adaptation response.
        """
        return self.step(e)


class Herzfeld14Model(ModelForRunner):
    def __init__(self, sigma=0.25, beta=0.01, alpha=1, basis_interval=[-3,3], n_basis_el=20, eta_ini=0.2, input_proj_vec=None, info=None):
        """
        Initialize the Herzfeld14 model parameters.
        """
        self.sigma = sigma
        self.beta = beta
        self.alpha = alpha
        self.basis_interval = basis_interval
        self.n_basis_el = n_basis_el
        self.eta_ini = eta_ini

        self.input_proj_vec = np.array(input_proj_vec) if input_proj_vec is not None else None
        self.reset_state()

    def reset_state(self):
        """Reset the model's adaptation states."""
        self.e_base = np.linspace(self.basis_interval[0], self.basis_interval[1], self.n_basis_el)
        self.w = np.ones(self.n_basis_el)
        
        g_at_zero = self.kernel_(0)
        eta_ = np.sum(self.w * g_at_zero)
        self.w = self.w / eta_ * self.eta_ini
        self.x = 0.0
        self.e_tm1 = 0.0

    def step(self, e: float) -> float:
        """
        Update the model's states given an error signal e.
        :param e: The error signal at the current step.
        :return: The total adaptation response.
        """
        if self.input_proj_vec is not None:
            e = np.dot(self.input_proj_vec, e)

        g = self.kernel_(e)
        eta = np.sum(self.w * g)
        self.w = self.w + self.beta * np.sign(e*self.e_tm1) * g / np.sum(g**2)
        self.e_tm1 = e
        return self.alpha * self.x + eta * e

    def kernel_(self, e):
        return np.exp(-(e-self.e_base)**2/(2*self.sigma**2))

    def __call__(self, e: float) -> float:
        """
        Allows the model to be called as model(e), equivalent to model.step(e).
        :param e: The error signal.
        :return: The total adaptation response.
        """
        return self.step(e)



class MLP_minimal(nn.Module, ModelForRunner):
    def __init__(self, n_inputs=None, n_hidden=None, n_outs=None, n_layers=1, nl='tanh', en_bias=True, prescaling=None, main_gain=None,
                 b_low=None, b_high=None, first_layer_init='default', skip_gain=None, first_layer_weights_trainable=False, out_layer_init='default',
                 info=None):
        super(MLP_minimal, self).__init__()
        

        nl = nl_selector(nl)
        self.activation = nl()

        # Input layer
        if b_low is None and b_high is not None:
            b_low = -b_high

        custom_first_bias = b_low is not None and b_high is not None

        self.n_layers = n_layers

        self.input_layer = nn.Linear(n_inputs, n_hidden, bias=en_bias or custom_first_bias)
        
        if first_layer_init == 'ones':
            nn.init.constant_(self.input_layer.weight, 1.0)          
        elif first_layer_init == 'uniform_unity':
            nn.init.uniform_(self.input_layer.weight, -1, 1)
        elif first_layer_init == 'default':
            pass
        else:
            raise ValueError('Unknown first_layer_init')

        self.input_layer.weight.requires_grad = first_layer_weights_trainable

        if custom_first_bias:
            bias_values = torch.linspace(b_low, b_high, n_hidden)
            with torch.no_grad():
                self.input_layer.bias.copy_(bias_values)
            self.input_layer.bias.requires_grad = False

        self.hidden_layers = nn.ModuleList([
            nn.Linear(n_hidden, n_hidden, bias=en_bias) for _ in range(n_layers - 1)
        ])

        self.output_layer = nn.Linear(n_hidden, n_outs, bias=en_bias)

        if out_layer_init == 'ones':
            nn.init.constant_(self.output_layer.weight, 1.0)
        elif out_layer_init == 'uniform_unity':
            nn.init.uniform_(self.output_layer.weight, -1, 1)
        elif out_layer_init == 'zeros':
            nn.init.constant_(self.output_layer.weight, 0.0)
        elif out_layer_init == 'default':
            pass

    def forward(self, x):
        
        x = self.input_layer(x)
        x = self.activation(x)

        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
        
        x = self.output_layer(x)

        return x
    
    def reset_state(self):
        pass