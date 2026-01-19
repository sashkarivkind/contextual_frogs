import sys
sys.path.append('/homes/ar2342/one_more_dir/contextual_frogs/experimental/optim/')
sys.path.append('/homes/ar2342/one_more_dir/contextual_frogs/')

import numpy as np
import torch
from types import SimpleNamespace
from models_part2 import BatchedElboGenerativeModelTop
import os
from optimise_clnn import load_subject_data

result_dir = '/homes/ar2342/one_more_dir/contextual_frogs/results_part2/hello_part2_tryMU3/'
os.makedirs(result_dir, exist_ok=True)
# -----------------------
# 1) Setup (match your routine)
# -----------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mode = 'MU' # 'ERSR' # 'MU'
paradigm_ = {k: 'evoked' if k <= 8 else 'spontaneous' for k in range(1, 17)}

n_seeds = 5  
n_subjects = 16 if mode == 'ERSR' else 24
n_epochs = 2000



args = SimpleNamespace(
    model='default',
    enable_q_scale_tuning= mode == 'MU',
    assume_opt_output_noise=True,
    enable_qlpf=False,
    enable_ylpf=False,
    enable_elpf=False,
    noise_injection_node='a',
    model_tie_lr_weight_decay=False,
    bs=n_subjects * n_seeds,                      # IMPORTANT: one batch entry per subject
    zzz_legacy_init=False,
    enable_output_scale_tuning= mode == 'MU',
    enable_u_feedback_scale_tuning=False,
    enable_direct_injection= mode == 'MU',
    injection_opt=2,            # you’re using opt=2 in the model code
    skip_gain=0.0,
    channel_trial_extra_error=0.0,
    n=128 if mode == 'ERSR' else 256,
)
# -----------------------
# 2) Load all 16 subjects, build [T, B] tensors (pad with NaNs)
# -----------------------
all_ys = []
all_a_exp = []
all_qs = []
lengths = []

for k in range(1, n_subjects + 1):
    if mode == 'ERSR':
        csv_path = f'/homes/ar2342/frogs_project/data/COIN_data/trial_data_{paradigm_[k]}_recovery_participant{(k-1)%8+1}.csv'
    else:
        csv_path = f'/homes/ar2342/frogs_project/data/COIN_data/trial_data_memory_updating_participant{k}.csv'
    experimental_data = load_subject_data(csv_path)

    a_exp = np.asarray(experimental_data[0], dtype=np.float32)  # target (your a_exp)
    ys    = np.asarray(experimental_data[1], dtype=np.float32)  # input ys (your ys)
    if mode == 'MU':
        qs = np.asarray(experimental_data[2], dtype=np.float32)  # input qs (your qs)

    for _ in range(n_seeds):
        all_a_exp.append(a_exp)
        all_ys.append(ys)
        if mode == 'MU':
            all_qs.append(qs)
        lengths.append(len(a_exp))

Tmax = int(max(lengths))
B = n_subjects * n_seeds

# pad to Tmax with NaN
ys_tb_np   = np.full((Tmax, B), np.nan, dtype=np.float32)
a_exp_tb_np = np.full((Tmax, B), np.nan, dtype=np.float32)
qs_tb_np = np.full((Tmax, B), 0.0, dtype=np.float32) if mode == 'MU' else None

for b in range(B):
    T = len(all_ys[b])
    ys_tb_np[:T, b]   = all_ys[b]
    a_exp_tb_np[:T, b] = all_a_exp[b]
    if mode == 'MU':
        qs_tb_np[:T, b] = all_qs[b]

ys_tb   = torch.tensor(ys_tb_np, device=device)     # [T,B]
a_exp_tb = torch.tensor(a_exp_tb_np, device=device)   # [T,B]
if mode == 'MU':
    qs_tb = torch.tensor(qs_tb_np, device=device)     # [T,B]

# -----------------------
# 3) Forward helper: produce a_pred [T,B] using the batched model
# -----------------------
def forward_tb(model, ys_tb, args, do_noise=False, qs_tb=None):
    """
    ys_tb: [T,B] with NaNs allowed
    Returns: a_pred_tb [T,B]
    """
    T, B = ys_tb.shape
    dev = next(model.parameters()).device

    ys_list = []
    qs_list = []
    if mode == 'MU':
        for t in range(T):
            q_t = qs_tb[t]  # [B]
            qs_list.append(q_t)
    for t in range(T):
        y_t = ys_tb[t]  # [B]
        # keep NaNs; model already handles NaNs internally
        ys_list.append(y_t)

    if do_noise:
        # NOTE: this injects *different* noise per subject, and grads flow through sigma_x (per subject)
        noises = [torch.randn(B, device=dev) * model.sigma_x for _ in range(T)]
    else:
        noises = [torch.zeros(B, device=dev) for _ in range(T)]

    a_list = model.f(
        n=args.n,
        noises=noises,         # list of [B]
        ys=ys_list,            # list of [B]
        model_setting=args.model,
        qs=qs_list if mode == 'MU' else None,
    )
    return torch.stack(a_list, dim=0)  # [T,B]

# -----------------------
# 4) Instantiate batched-parameter model and train on all 16 subjects at once
# -----------------------
# You must have BatchedElboGenerativeModelTop defined from the earlier response.
model = BatchedElboGenerativeModelTop(device=device, args=args, batch_size=args.bs).to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(n_epochs):
    model.train()
    opt.zero_grad()

    a_pred_tb = forward_tb(model, ys_tb, args, do_noise=False, qs_tb=qs_tb)  # [T,B]

    # loss: mean over subjects of per-subject MSE over available timepoints
    # mask = ~torch.isnan(a_exp_tb)                                # [T,B]
    # se = (a_pred_tb - a_exp_tb) ** 2
    # se = torch.where(mask, se, torch.zeros_like(se))

    # counts = mask.sum(dim=0).clamp_min(1)                       # [B]
    # mse_per_subject = se.sum(dim=0) / counts                    # [B]
    # loss = mse_per_subject.mean()

    mask = ~torch.isnan(a_exp_tb)                           # [T,B] bool
    mask_f = mask.to(a_pred_tb.dtype)                      # [T,B] float

    a_exp_filled = torch.nan_to_num(a_exp_tb, nan=0.0)        # [T,B], no NaNs
    diff = (a_pred_tb - a_exp_filled) * mask_f              # [T,B], masked; no NaNs

    se_sum = (diff * diff).sum(dim=0)                      # [B]
    counts = mask.sum(dim=0).clamp_min(1)                  # [B]
    mse_per_subject = se_sum / counts                      # [B]
    loss = mse_per_subject.mean()                          # scalar

    loss.backward()
    opt.step()

    if (epoch % 10) == 0 or epoch == n_epochs - 1:
        rmse_per_subject = torch.sqrt(mse_per_subject).detach().cpu().numpy().reshape(n_subjects, n_seeds)
        best_rmse = rmse_per_subject.min(axis=1)
        print(
            f"Epoch {epoch:04d} | "
            f"loss(mean MSE)={loss.item():.6g} | "
            f"rmse(mean)={rmse_per_subject.mean():.6g} | "
            f"rmse(per-subject)=\n {np.round(rmse_per_subject, 4)} | "
            f"best_rmse(per-subject)=\n {np.round(best_rmse, 4)}"
        )

# -----------------------
# 5) Evaluate final model on all subjects, save a vecotr of predictions and of model outputs for each subject and seed
# -----------------------
with torch.no_grad():
    model.eval()
    a_pred_tb = forward_tb(model, ys_tb, args, do_noise=False, qs_tb=qs_tb)  # [T,B]

    # save predictions
    a_pred_tb_cpu = a_pred_tb.detach().cpu().numpy()  # [T,B]
    np.savetxt(result_dir + 'a_exp.txt', a_exp_tb_np)
    np.savetxt(result_dir + 'a_pred.txt', a_pred_tb_cpu)


# -----------------------
# 6) Outcome: batched optimised parameters (each is [B])
# -----------------------
with torch.no_grad():
    print("\nOptimised per-subject params (each is [16]):")
    print("sigma_b:", model.sigma_b.detach().cpu().numpy())
    print("sp_weight_decay:", model.sp_weight_decay.detach().cpu().numpy())
    print("log_learning_rate:", model.log_learning_rate.detach().cpu().numpy())
    print("log_learning_rate_decay:", model.log_learning_rate_decay.detach().cpu().numpy())
    if isinstance(model.output_scale, torch.Tensor) and model.output_scale.requires_grad:
        print("output_scale:", model.output_scale.detach().cpu().numpy())
    if isinstance(model.u_feedback_scale, torch.Tensor) and model.u_feedback_scale.requires_grad:
        print("u_feedback_scale:", model.u_feedback_scale.detach().cpu().numpy())
    if isinstance(model.q_scale, torch.Tensor) and model.q_scale.requires_grad:
        print("q_scale:", model.q_scale.detach().cpu().numpy())
    if isinstance(model.direct_injection_scale, torch.Tensor) and model.direct_injection_scale.requires_grad:
        print("direct_injection_scale:", model.direct_injection_scale.detach().cpu().numpy())
    # -----------------------


# save best per-subject parameters
    np.savetxt(result_dir + 'sigma_b.txt', model.sigma_b.detach().cpu().numpy())
    np.savetxt(result_dir + 'sp_weight_decay.txt', model.sp_weight_decay.detach().cpu().numpy())
    np.savetxt(result_dir + 'log_learning_rate.txt', model.log_learning_rate.detach().cpu().numpy())
    np.savetxt(result_dir + 'log_learning_rate_decay.txt', model.log_learning_rate_decay.detach().cpu().numpy())
    if isinstance(model.output_scale, torch.Tensor) and model.output_scale.requires_grad:
        np.savetxt(result_dir + 'output_scale.txt', model.output_scale.detach().cpu().numpy())
    if isinstance(model.u_feedback_scale, torch.Tensor) and model.u_feedback_scale.requires_grad:
        np.savetxt(result_dir + 'u_feedback_scale.txt', model.u_feedback_scale.detach().cpu().numpy())
    if isinstance(model.q_scale, torch.Tensor) and model.q_scale.requires_grad:
        np.savetxt(result_dir + 'q_scale.txt', model.q_scale.detach().cpu().numpy())
    if isinstance(model.direct_injection_scale, torch.Tensor) and model.direct_injection_scale.requires_grad:
        np.savetxt(result_dir + 'direct_injection_scale.txt', model.direct_injection_scale.detach().cpu().numpy())