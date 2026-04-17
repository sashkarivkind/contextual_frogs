'''utilities for handling, evaluating, and playingi with models'''
from xml.parsers.expat import model
import torch
import numpy as np
from collections.abc import Mapping
from collections import OrderedDict
from typing import MutableMapping

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

# def force_model_params(model, forced_params):
#     for name, param in model.named_parameters():
#         if name in forced_params:
#             print(f'Forcing parameter {name} to value {forced_params[name]}')
#             param.data.fill_(forced_params[name])

def fwd_pass(model, ys, args, do_noise=False, qs=None):
    if not do_noise:
            noises = [torch.zeros((args.bs,), device=next(model.parameters()).device) for _ in range(len(ys))]  # [bs, t]
    else:
            raise NotImplementedError("Noise injection not implemented in this snippet")
    model_setting = args.model
    outputs_ = model.f(args.n,
                            noises,
                            ys,  
                            model_setting,
                            qs=qs,
                            )
    outputs = torch.stack(outputs_)
    return outputs.mean(axis=1)

def force_model_params(model, forced_params):
    for name, param in model.named_parameters():
        if name not in forced_params:
            continue

        value = forced_params[name]
        print(f"Forcing parameter {name} to value {value}")

        with torch.no_grad():
            value = torch.as_tensor(value, dtype=param.dtype, device=param.device)

            # Scalar case
            if value.ndim == 0:
                param.fill_(value.item())
                continue

            # Exact shape match
            if value.shape == param.shape:
                param.copy_(value)
                continue

            # Allow broadcastable shapes but check explicitly
            try:
                broadcasted = value.expand(param.shape)
            except RuntimeError:
                raise ValueError(
                    f"Shape mismatch when forcing '{name}': "
                    f"parameter shape {tuple(param.shape)} vs value shape {tuple(value.shape)}"
                )

            param.copy_(broadcasted)

@torch.no_grad()
def migrate_output_scale_to_input_scale_state_dict(
    state_dict: MutableMapping[str, torch.Tensor],
    *,
    inplace: bool = False,
    prefix: str = "",
    remove_output_scale: bool = True,
) -> MutableMapping[str, torch.Tensor]:
    """
    Remap an old checkpoint/state_dict from:

        a_mean = output_scale * u

    to:

        y_input = input_scale * y
        a_mean = u

    while preserving the full ReLU dynamics under:
      - enable_w_in_plasticity = False
      - enable_bias_update = False
      - nl_activation = "relu"
      - usual learning exponent = 1

    Required old keys:
      - output_scale
      - sigma_b
      - log_learning_rate

    Optional old keys:
      - q_scale

    Adds/replaces:
      - input_scale

    Sets:
      - output_scale = 1

    Use `prefix="module."` for DataParallel/DDP-style checkpoints.
    """

    if inplace:
        sd = state_dict
    else:
        sd = OrderedDict(
            (k, v.clone() if torch.is_tensor(v) else v)
            for k, v in state_dict.items()
        )

    output_scale_key = prefix + "output_scale"
    input_scale_key = prefix + "input_scale"
    sigma_b_key = prefix + "sigma_b"
    q_scale_key = prefix + "q_scale"
    log_lr_key = prefix + "log_learning_rate"

    required = [output_scale_key, sigma_b_key, log_lr_key]
    missing = [k for k in required if k not in sd]
    if missing:
        raise KeyError(f"Missing required state_dict keys: {missing}")

    c = sd[output_scale_key].detach().clone()

    if torch.any(c <= 0):
        raise ValueError(
            "Exact ReLU remap requires strictly positive output_scale values."
        )

    if c.ndim != 1:
        raise ValueError(
            f"Expected output_scale to have shape [bs], got {tuple(c.shape)}"
        )

    bs = c.shape[0]

    if sd[sigma_b_key].shape[0] != bs:
        raise ValueError(
            f"{sigma_b_key} batch dimension does not match output_scale."
        )

    if sd[log_lr_key].shape[0] != bs:
        raise ValueError(
            f"{log_lr_key} batch dimension does not match output_scale."
        )

    # New input-side scale.
    sd[input_scale_key] = c.clone()

    # Disable output-side scale.
    if remove_output_scale:
        del sd[output_scale_key]
    else:
        sd[output_scale_key] = torch.ones_like(c)

    # Scale hidden preactivation bias term.
    sd[sigma_b_key] = sd[sigma_b_key] * c

    # Scale q input term if present.
    if q_scale_key in sd:
        sd[q_scale_key] = sd[q_scale_key] * c

    # Preserve plastic learning dynamics:
    # lr_new = lr_old / c**2
    sd[log_lr_key] = sd[log_lr_key] - 2.0 * torch.log(c)[:, None]

    return sd