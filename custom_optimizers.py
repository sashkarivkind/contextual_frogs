import torch
from torch.optim.optimizer import Optimizer, required

class ElementWiseDecay(Optimizer):
    r"""
    Implements a custom optimizer where each weight w_ij maintains its own learning rate
    multiplier that decays based on the magnitude of the weight update.
    
    For each weight element w_ij, let:
    
        effective_lr_ij = base_lr * lr_mult_ij
        
    The weight update is then:
    
        delta_w_ij = effective_lr_ij * grad_ij
    
    and then we decay the multiplier:
    
        lr_mult_ij = lr_mult_ij * exp(-alpha * |delta_w_ij|)
    
    where alpha is a hyperparameter (alpha > 0) controlling the influence of the update magnitude.
    
    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Base learning rate.
        alpha (float): Decay hyperparameter that scales the influence of the absolute update magnitude.
                       A typical value might be small (e.g. 0.001 to 0.1) to avoid decaying too fast.
    """
    def __init__(self, params, lr=required, alpha=0.01):
        defaults = dict(lr=lr, alpha=alpha)
        super(ElementWiseDecay, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            base_lr = group['lr']
            alpha = group['alpha']
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Use the .data attribute to work with tensors directly.
                grad = p.grad.data
                # Initialize per-element learning rate multipliers if necessary.
                state = self.state[p]
                if 'lr_mult' not in state:
                    # Initialize with ones of the same shape as the parameter.
                    state['lr_mult'] = torch.ones_like(p.data)
                
                # effective learning rate for each weight element
                lr_mult = state['lr_mult']
                effective_lr = base_lr * lr_mult  # elementwise multiplication
                
                # Compute the update for each element.
                update = effective_lr * grad
                
                # Update the parameter.
                p.data.add_(-update)
                
                # Decay the learning rate multipliers using the absolute update.
                # We use an exponential decay function: new_lr_mult = lr_mult * exp(-alpha * |update|)
                decay_factor = torch.exp(-alpha * torch.abs(update))
                state['lr_mult'].mul_(decay_factor)
                
        return loss
