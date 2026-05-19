import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

def compute_k_params(model_state_dict, 
                     exclude_by_startwith=['_'], 
                     exclude_by_name=['sigma_x'], 
                     count_for_one={'sigma_b', 'direct_injection_scale','output_scale','u_feedback_scale', 
                                    'x_slow_alpha', 'x_fast_gain', 'x_fast_alpha'}):
    k_params = 0
    for key in model_state_dict:
        if any([key.startswith(excl) for excl in exclude_by_startwith]):
            continue
        if key in exclude_by_name:
            continue
        param_tensor = model_state_dict[key]
        this_param_size = param_tensor.shape[-1] if key not in count_for_one else 1
        print(f'param: {key}, size: {this_param_size}')
        k_params += this_param_size
    return k_params

def read_rmse_and_bic_from_path(bic_path,seeds):
    model_state_dict = torch.load(os.path.join(bic_path, 'model_state_dict.pt'), map_location='cpu')
    k_params = compute_k_params(model_state_dict)
    print(f'detected k_params: {k_params}')
    a_exp = np.loadtxt(os.path.join(bic_path, 'a_exp.txt'))
    a_pred = np.loadtxt(os.path.join(bic_path, 'a_pred.txt'))

    subjs = a_exp.shape[1] // seeds
    print(f'detected subjs: {subjs}')

    rmse = np.sqrt(np.nanmean((a_exp - a_pred) ** 2,axis=0)).reshape([subjs, seeds])
    n_samples = np.sum(~np.isnan(a_exp),axis=0).reshape([subjs, seeds])
    assert np.all(n_samples == n_samples[:, [0]])
    n_samples = n_samples[:, 0]

    best_seed_idx = np.nanargmin(rmse, axis=1)
    best_rmse = rmse[np.arange(subjs), best_seed_idx]
    best_bics = [bic_gaussian_from_rmse(this_rmse, n=this_n, k=k_params) for this_rmse, this_n in zip(best_rmse, n_samples)]
    return {"best_rmse": np.array(best_rmse), "best_bics": np.array(best_bics)}

def eval_single_paradigm(model, 
                         paradigm, 
                         args, 
                         paradigm_name=None, 
                         record_internals=False, 
                         record_vectors=False,
                         noise_spikes=[]):
    if paradigm_name is not None:
        print(f"evaluating paradigm {paradigm_name} with {len(paradigm)} steps")
    ys = torch.tensor(paradigm, device=next(model.parameters()).device)
    # duplicate ys across batch size
    ys = ys.unsqueeze(0).repeat(args.bs, 1)  # [bs, t]
    # noises = torch.randn((args.bs, ys.shape[1]), device=next(model.parameters()).device) * model.sigma_x  # [bs, t]
    noises = torch.zeros((args.bs, ys.shape[1]), device=next(model.parameters()).device)  # [bs, t]
    for t in noise_spikes:
        if t < noises.shape[1]:
            noises[:, t] = 1

    noises = [noises[:, t_idx] for t_idx in range(noises.shape[1])]
    ys = [ys[:, t_idx] for t_idx in range(ys.shape[1])]
    if paradigm_name is not None:
        print(f"noises shape: {noises[0].shape}, ys shape: {ys[0].shape}")
    model_setting = args.model
    outputs_ = model.f(
        args.n,
        noises,
        ys,
        model_setting,
        qs=None,
        record_internals=record_internals,
        record_vectors=record_vectors,
    )
    if record_internals:
        outputs_, internals = outputs_
        return outputs_, internals
    return outputs_, None


def local_eval_paradigms(model, playlist_file, args, record_internals=False, record_vectors=False):
    with open(playlist_file, "rb") as f:
        playlist = pickle.load(f)

    model.eval()
    with torch.no_grad():
        outputs = {}
        internals_by_paradigm = {} if record_internals else None
        for paradigm_name, paradigm in playlist.items():
            outputs_, internals = eval_single_paradigm(
                model,
                paradigm,
                args,
                paradigm_name=paradigm_name,
                record_internals=record_internals,
                record_vectors=record_vectors,
            )
            outputs[paradigm_name] = np.array([z.cpu().numpy().reshape(-1) for z in outputs_])
            if record_internals:
                internals_by_paradigm[paradigm_name] = internals
    if record_internals:
        return outputs, internals_by_paradigm
    return outputs

def eval_single_wrap(model, paradigm, args, paradigm_name=None, record_internals=False, record_vectors=False):
    outputs, internals = eval_single_paradigm(model, paradigm, args, paradigm_name, record_internals, record_vectors)
    outputs = [o.detach().cpu().numpy() for o in outputs]
    outputs = np.array(outputs)
    if record_internals:
        for k in internals:
            internals[k] = {kk: vv.detach().cpu().numpy() for kk, vv in internals[k].items()}
        return outputs, internals
    return outputs, None

def find_fixed_points(x_values, y_values, tol=1e-3, selection='closest_to_zero'):
    x_values = np.asarray(x_values, dtype=np.float32)
    y_values = np.asarray(y_values, dtype=np.float32)

    if y_values.ndim == 1:
        y_values = y_values[None, :]

    fixed_points = []
    for y in y_values:
        diffs = y - x_values
        candidates = []

        close_idx = np.where(np.abs(diffs) <= tol)[0]
        candidates.extend(x_values[close_idx].tolist())

        sign_changes = np.where(np.signbit(diffs[:-1]) != np.signbit(diffs[1:]))[0]
        for idx in sign_changes:
            x0, x1 = x_values[idx], x_values[idx + 1]
            d0, d1 = diffs[idx], diffs[idx + 1]
            # if np.isclose(d0, d1):
            #     crossing = 0.5 * (x0 + x1)
            # else:
            crossing = x0 - d0 * (x1 - x0) / (d1 - d0)
            candidates.append(float(crossing))

        if not candidates:
            fixed_points.append(np.nan)
            continue

        candidates = np.array(sorted(set(np.round(candidates, 8))), dtype=np.float32)
        if selection == 'closest_to_zero':
            fixed_points.append(candidates[np.argmin(np.abs(candidates))])
        elif selection == 'first':
            fixed_points.append(candidates[0])
        elif selection == 'last':
            fixed_points.append(candidates[-1])
        else:
            raise ValueError(f'Unknown selection mode: {selection}')

    return np.asarray(fixed_points, dtype=np.float32)

def eval_paradigm_with_inoutmaps(model, args, paradigm, probe_values, *, name=None):
    if name is not None:
        print(f"evaluating {name} with {len(paradigm)} steps")

    device = next(model.parameters()).device
    ys_tensor = torch.tensor(paradigm, device=device).unsqueeze(0).repeat(args.bs, 1)
    noise_tensor = torch.zeros_like(ys_tensor)
    ys_list = [ys_tensor[:, t_idx] for t_idx in range(ys_tensor.shape[1])]
    noise_list = [noise_tensor[:, t_idx] for t_idx in range(noise_tensor.shape[1])]

    outputs = None
    inoutmaps = []

    model.eval()
    with torch.no_grad():
        for probe_value in probe_values:
            probing_vec = torch.full((args.bs,), float(probe_value), device=device)
            outputs_raw, inoutmaps_raw = model.f(
                args.n,
                noise_list,
                ys_list,
                args.model,
                qs=None,
                record_inoutmaps=True,
                inoutmaps_probing_vec=probing_vec,
            )

            if outputs is None:
                outputs = np.stack([
                    z.detach().cpu().numpy().reshape(-1) for z in outputs_raw
                ], axis=0)

            inoutmaps.append(np.stack([
                hidden.detach().cpu().numpy().reshape(-1) for hidden, _ in inoutmaps_raw
            ], axis=0))

    inoutmaps = np.stack(inoutmaps, axis=0)
    inoutmaps = np.moveaxis(inoutmaps, 0, -1)
    fixed_points = np.stack([
        find_fixed_points(probe_values, inoutmaps[:, subj_idx, :])
        for subj_idx in range(outputs.shape[1])
    ], axis=1)

    trial_idx = np.arange(len(paradigm))
    output_mean = outputs.mean(axis=1)
    fixed_point_mean = np.nanmean(fixed_points, axis=1)

    return {
        'name': name,
        'paradigm': paradigm,
        'outputs': outputs,
        'output_mean': output_mean,
        'inoutmaps': inoutmaps,
        'fixed_points': fixed_points,
        'fixed_point_mean': fixed_point_mean,
        'trial_idx': trial_idx,
    }

def myplot(data, label=None,align_by_endpoint=True, subj=[0,16], add_sem=True, pooling_fun=np.mean):
    if align_by_endpoint:
        x_data = np.arange(data.shape[0]) - (data.shape[0]-1)
    else:
        x_data = np.arange(data.shape[0])
    y_data = data[:, subj[0]:subj[1]]
    plt.plot(x_data, pooling_fun(y_data, axis=1), label=label)
    if add_sem:
        n = np.sum(~np.isnan(y_data), axis=1)
        sem = np.divide(
            np.nanstd(y_data, axis=1, ddof=1),
            np.sqrt(n),
            out=np.zeros_like(x_data, dtype=float),
            where=n > 1,
        )
        plt.fill_between(
            x_data,
            pooling_fun(y_data, axis=1) - sem,
            pooling_fun(y_data, axis=1) + sem,
            alpha=0.2,
            linewidth=0,
        )

