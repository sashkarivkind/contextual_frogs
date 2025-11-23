'''utilities for handling, evaluating, and playingi with models'''
from xml.parsers.expat import model
import torch
import numpy as np
from collections.abc import Mapping

def eval_ys(model, ys, args, manual_noises=None, qs=None):
    model.eval()
    with torch.no_grad():
            if manual_noises is not None:
                noises = manual_noises
            else:
                noises = torch.randn((args.bs, len(ys)), device=next(model.parameters()).device) * model.sigma_x  # [bs, t]
                noises = [noises[:, t_idx] for t_idx in range(noises.shape[1])]
            model_setting = args.model
            outputs_ = model.f(args.n,
                                noises,
                                ys,  
                                model_setting,
                                qs=qs,
                                )
    return np.array([z.cpu().numpy().reshape(-1) for z in outputs_]) 



def remove_gen(row_params: Mapping, prefix: str = "gen", require: bool = True) -> dict:
    """
    Filter a flat state_dict-like mapping to only <prefix>.* keys and strip the '<prefix>.' part.

    Args:
        row_params: Mapping of parameter names -> tensors (e.g., a state_dict row).
        prefix:     Top-level prefix to keep (default: 'gen').
        require:    If True, raise a ValueError when no <prefix>. keys are found.

    Returns:
        A new dict suitable for model.load_state_dict(...), with the prefix removed.
    """
    pre = f"{prefix}."
    out = {k[len(pre):]: v for k, v in row_params.items() if isinstance(k, str) and k.startswith(pre)}
    if require and not out:
        raise ValueError(f"No keys found starting with '{pre}'")
    return out

def nans2none(arr):
    new_list = []
    for a in arr:
        if np.isnan(a):
            new_list.append(None)
        else:
            new_list.append(a)
    return new_list

def force_model_params(model, forced_params):
    for name, param in model.named_parameters():
        if name in forced_params:
            print(f'Forcing parameter {name} to value {forced_params[name]}')
            param.data.fill_(forced_params[name])