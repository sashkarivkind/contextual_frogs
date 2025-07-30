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
    
        lr_mult_ij = lr_mult_ij * exp(-alpha * |delta_w_ij|**beta)
    
    where alpha is a hyperparameter (alpha > 0) controlling the influence of the update magnitude.
    if beta is not specified (equals None) then it is set to 1

    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Base learning rate.
        alpha (float): Decay hyperparameter that scales the influence of the absolute update magnitude.
                       A typical value might be small (e.g. 0.001 to 0.1) to avoid decaying too fast.
    """
    def __init__(self, params, lr=required, alpha=0.01, min_lr_mult=0.0, beta = None):
        defaults = dict(lr=lr, alpha=alpha, min_lr_mult=min_lr_mult, beta=beta)
        super(ElementWiseDecay, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            base_lr = group['lr']
            alpha = group['alpha']
            min_lr_mult = group['min_lr_mult']

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
                # We use an exponential decay function: new_lr_mult = lr_mult * exp(-alpha * |update| ** beta)
                if group['beta'] is not None:
                    decay_factor = torch.exp(-alpha * torch.abs(update) ** group['beta'])
                else:
                    decay_factor = torch.exp(-alpha * torch.abs(update))
                state['lr_mult'].mul_(decay_factor)
                # Ensure that lr_mult does not go below the minimum threshold.
                state['lr_mult'].clamp_(min=min_lr_mult)
                
        return loss
    
        def get_lr_mult(self, param):
            #TODO: test this method (if it goes into any model)
            """Get the current learning rate multiplier for a specific parameter."""
            return self.state[param].get('lr_mult', torch.ones_like(param.data))


class GlobalPNormDecay(Optimizer):
    r"""
    Implements a custom optimizer where a single global learning‐rate multiplier
    decays based on the p‐norm of the parameter updates.

    For each step:
        1. Compute the p‐norm of the raw updates: 
               ‖Δ‖_p = ( ∑_i |base_lr * grad_i|^p )^(1/p)
        2. Apply the update with an effective learning rate:
               effective_lr = base_lr * lr_mult
               θ_i ← θ_i − effective_lr * grad_i - wieght_decay * θ_i        
        3. Decay the global multiplier:
               lr_mult ← lr_mult * exp(−α * ‖Δ‖_p)
        4. Clamp lr_mult ≥ min_lr_mult

    Arguments:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):        Base learning rate.
        alpha (float):     Decay hyperparameter (α > 0).
        p (float, optional): Norm degree for pulling the updates (default: 1).
        min_lr_mult (float): Minimum allowed global multiplier (default: 0).
        weight_decay (float): Weight decay factor (default: 0.0).
        tau_relax (float, optional): If specified, relaxes the global multiplier based on
                                     the tau_relax value as log_lr_mult = log(lr_mult)*(1 - tau_relax).
    """
    def __init__(self, params, lr=required, alpha=0.01, p=1.0, min_lr_mult=0.0, tau_relax=None, weight_decay=0.0):
        defaults = dict(lr=lr, alpha=alpha, p=p, min_lr_mult=min_lr_mult,tau_relax=tau_relax, weight_decay=weight_decay)
        super(GlobalPNormDecay, self).__init__(params, defaults)
        # initialize a global multiplier for each parameter group
        for group in self.param_groups:
            group['lr_mult'] = 1.0

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            base_lr       = group['lr']
            alpha         = group['alpha']
            p_norm        = group['p']
            min_lr_mult   = group['min_lr_mult']
            lr_mult       = group.get('lr_mult', 1.0)
            weight_decay  = group['weight_decay']
            tau_relax     = group['tau_relax'] if 'tau_relax' in group else None

            # --- 1) compute the p‐norm of the raw updates (base_lr * grad) ---
            if p_norm == float('inf'):
                # infinity norm: maximum absolute update
                max_vals = []
                for param in group['params']:
                    if param.grad is None:
                        continue
                    max_vals.append((base_lr * param.grad.data).abs().max())
                if max_vals:
                    update_norm = torch.stack(max_vals).max()
                else:
                    update_norm = torch.tensor(0.0, device=param.device)
            else:
                total = torch.tensor(0.0, device=group['params'][0].device)
                for param in group['params']:
                    if param.grad is None:
                        continue
                    total = total + (base_lr * param.grad.data).abs().pow(p_norm).sum()
                update_norm = total.pow(1.0 / p_norm)

            # --- 2) apply the parameter updates with the effective LR ---
            effective_lr = base_lr * lr_mult
            for param in group['params']:
                if param.grad is None:
                    continue
                param.data.add_(-effective_lr * param.grad.data - lr_mult * weight_decay * param.data)

            # --- 3) decay the global learning‐rate multiplier ---
            decay = torch.exp(-alpha * update_norm)
            new_lr_mult = lr_mult * float(decay)

            # --- 4) clamp to minimum and store back in group ---
            group['lr_mult'] = max(min_lr_mult, new_lr_mult)

            # --- 5) if tau_relax is specified, apply it to the global multiplier ---
            if tau_relax is not None:
                # Relax the multiplier based on the tau_relax value as log_lr_mult = log(lr_mult)*(1 - tau_relax)
                log_lr_mult = torch.log(group['lr_mult'])
                log_lr_mult = log_lr_mult * (1 - tau_relax)
                group['lr_mult'] = torch.exp(log_lr_mult).item()
                
        return loss

    def get_global_lr_mult(self):
        """Ensure that the lr_mult is consistent across all parameter groups."""
        if len(self.param_groups) == 0:
            return 1.0
        else: #sweep through all the groups ensure same lr_mult if yes - return it; else raise an error
            first_lr_mult = self.param_groups[0]['lr_mult']
            for group in self.param_groups:
                if group['lr_mult'] != first_lr_mult:
                    raise ValueError("get_global_lr_mult called but lr_mult is not consistent across parameter groups.")
            # If all groups have the same lr_mult, return it.
            return first_lr_mult
    
    def get_global_lr(self):
        """Get the current global learning rate."""
        #sweep through all the groups ensure same lr_mult if yes - return lr_mult*lr; else raise an error
        if len(self.param_groups) == 0:
            return 1.0
        else:
            first_lr_mult = self.param_groups[0]['lr_mult']
            for group in self.param_groups:
                if group['lr_mult'] != first_lr_mult:
                    raise ValueError("get_global_lr called but lr_mult is not consistent across parameter groups.")
            # If all groups have the same lr_mult, return it.
            return first_lr_mult * self.param_groups[0]['lr']
