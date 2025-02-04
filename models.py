import torch
import torch.nn as nn

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


class MLP(nn.Module):
    def __init__(self, n_inputs=None, n_hidden=None, n_outs=None, n_layers=1, nl='tanh', en_bias=True,
                 b_low=None, b_high=None, first_layer_init='default', skip_gain=None, first_layer_weights_trainable=False):
        super(MLP, self).__init__()

        # Input layer
        if b_low is None and b_high is not None:
            b_low = -b_high

        custom_first_bias = b_low is not None and b_high is not None

        self.n_layers = n_layers
        self.skip_gain = skip_gain if skip_gain is not None else 0  # Default gain is 0

        nl = nl_selector(nl)
        self.activation = nl()

        # Define layers explicitly
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

    def forward(self, x):

        x_res = torch.sum(x, dim=-1, keepdim=True)

        x = self.input_layer(x)
        x = self.activation(x)

        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
        
        x = self.output_layer(x) + self.skip_gain * x_res  
        return x


class SymReLU(torch.nn.Module):
    def __init__(self, threshold=1):
        super(SymReLU, self).__init__()
        self.threshold = threshold
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.threshold - torch.abs(x))


class OneOverSqr(torch.nn.Module):
    def __init__(self, w=1, c=0):
        super(OneOverSqr, self).__init__()
        self.w = w
        self.c = c

    def forward(self, x):
        return (1 + (x / self.w).pow(2)).reciprocal() + self.c