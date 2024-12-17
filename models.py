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

# Define the MLP Model
class MLP(nn.Module):
    def __init__(self, n_inputs=None, n_hidden=None, n_outs=None, n_layers=1, nl='tanh', en_bias=True,
                 b_low=None, b_high=None, ones_first_layer=False):
        """
        Args:
            n_inputs (int): Number of input neurons
            n_hidden (int): Number of hidden neurons
            n_outs (int): Number of output neurons
            n_layers (int): Number of layers
            nl (str): Non-linearity ('tanh', 'relu', 'sigmoid')
            en_bias (bool): Enable bias in layers
            b_low (float): Lower bound for bias linspace in the first layer (optional)
            b_high (float): Upper bound for bias linspace in the first layer (optional)
            ones_first_layer (bool): Initialize first layer weights as all ones
            custom_first_bias (bool): Enable custom bias initialization in the first layer
        """
        super(MLP, self).__init__()

        nl = nl_selector(nl)
        layers = []

        # Input layer
        if b_low is None and b_high is not None:
            b_low = -b_high

        custom_first_bias = b_low is not None and b_high is not None

        first_layer = nn.Linear(n_inputs, n_hidden, bias=en_bias or custom_first_bias)
        if ones_first_layer:
            # Initialize input weights as all ones
            nn.init.constant_(first_layer.weight, 1.0)
            first_layer.weight.requires_grad = False
        if custom_first_bias:
            # Set bias values using linspace between b_low and b_high
            bias_values = torch.linspace(b_low, b_high, n_hidden)
            with torch.no_grad():
                first_layer.bias.copy_(bias_values)
            first_layer.bias.requires_grad = False

        layers.append(first_layer)
        layers.append(nl())

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden, bias=en_bias))
            layers.append(nl())

        # Output layer
        layers.append(nn.Linear(n_hidden, n_outs, bias=en_bias))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    


class SymReLU(torch.nn.Module):
    def __init__(self, threshold=1):
        super(SymReLU, self).__init__()
        self.threshold = threshold
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.threshold - torch.abs(x))


class OneOverSqr(torch.nn.Module):
    def __init__(self, w=1):
        super(OneOverSqr, self).__init__()
        self.w = w

    def forward(self, x):
        return (1 + (x / self.w).pow(2)).reciprocal()