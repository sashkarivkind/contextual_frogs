import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import quad, dblquad, tplquad

def compute_ntk_matrix_np(model, x_vec, device="cpu"):
    """Compute the NTK matrix for a given model and input vector, supporting CUDA if available."""
    model.to(device)
    if not torch.is_tensor(x_vec):
        x_vec = torch.tensor(x_vec, dtype=torch.float32)
    x_vec = x_vec.to(device)

    # Collect the model parameters
    params = list(model.parameters())

    # Initialize the NTK matrix as a NumPy array
    ntk_matrix = np.zeros((len(x_vec), len(x_vec)))

    for i, x_i in enumerate(x_vec):
        # Compute output and gradients for x_i
        model.zero_grad()
        y_i = model(x_i)
        y_i.backward(retain_graph=True)

        # Flatten the gradients into a vector and convert to numpy
        jacobian_i = torch.cat([p.grad.view(-1) for p in params if p.grad is not None]).detach().cpu().numpy()

        for j, x_j in enumerate(x_vec):
            if i <= j:  # NTK is symmetric; compute only for i <= j
                model.zero_grad()
                y_j = model(x_j)
                y_j.backward(retain_graph=True)
                jacobian_j = torch.cat([p.grad.view(-1) for p in params if p.grad is not None]).detach().cpu().numpy()

                # Compute the kernel value K(x_i, x_j) as dot product of Jacobians (in numpy)
                ntk_value = np.dot(jacobian_i, jacobian_j)
                ntk_matrix[i, j] = ntk_value
                ntk_matrix[j, i] = ntk_value  # Symmetry

    return ntk_matrix


def compute_dntk_dxj_tensor_np(model, x_vec, device="cpu"):
    """
    Compute the gradient of the NTK with respect to the second input.
    
    Given a model f with parameters θ, the NTK is
        K(x_i, x_j) = ∇_θ f(x_i) · ∇_θ f(x_j).
    This function computes
        dK/dx_j (a vector in R^D)
    for each pair (i, j), returning a numpy array of shape (N, N, D).

    Args:
        model      : a torch.nn.Module mapping R^D → R (scalar output).
        x_vec      : torch.Tensor of shape (N, D).
        device     : "cpu" or "cuda".

    Returns:
        dntk       : np.ndarray of shape (N, N, D) where
                     dntk[i, j, :] = ∂K(x_i, x_j) / ∂x_j.
    """
    model.to(device)
    if not torch.is_tensor(x_vec):
        x_vec = torch.tensor(x_vec, dtype=torch.float32)
    x_vec = x_vec.to(device)
    params = list(model.parameters())
    #filter out parameters that do not require grad
    params = [p for p in params if p.requires_grad]
    N, D = x_vec.shape

    # Allocate storage
    dntk = np.zeros((N, N, D), dtype=np.float32)

    # Precompute ∇_θ f(x_i) for all i
    grads_wrt_params = []
    for i in range(N):
        model.zero_grad()
        y_i = model(x_vec[i])
        # grad_i: list of parameter gradients ∇_θ f(x_i)
        grad_i = torch.autograd.grad(y_i, params, retain_graph=True)
        # flatten into one vector
        grad_i_vec = torch.cat([g.contiguous().view(-1) for g in grad_i]).detach()
        grads_wrt_params.append(grad_i_vec)

    # Now for each j, compute ∂/∂x_j K(x_i, x_j)
    for j in range(N):
        # detach and enable grad on x_j
        xj = x_vec[j].clone().detach().to(device).requires_grad_(True)

        # compute ∇_θ f(x_j) with graph
        model.zero_grad()
        y_j = model(xj)
        # print(' 2.   y_j requires grad:', y_j.requires_grad)
        grad_j = torch.autograd.grad(y_j, params, create_graph=True)
        grad_j_vec = torch.cat([g.contiguous().view(-1) for g in grad_j])

        # for each i, form K_ij and backprop to xj
        for i in range(N):
            grad_i_vec = grads_wrt_params[i]

            # scalar K_ij = grad_i ⋅ grad_j
            K_ij = torch.dot(grad_i_vec, grad_j_vec)

            # compute ∇_{x_j} K_ij
            dK_dxj = torch.autograd.grad(K_ij, xj, retain_graph=True)[0]

            dntk[i, j, :] = dK_dxj.detach().cpu().numpy()

    return dntk


def compute_ntk_ana(Nin=None,N=None,P0=None,Pplus=None,nl=None,nl_p=None, return_components=False):
    sigmoid = lambda x: 1./(1+np.exp(-x))
    sigmoid_p = lambda x: sigmoid(x) * (1 - sigmoid(x))

    tanh = np.tanh
    tanh_p = lambda x: 1 - tanh(x)**2

    w1h = b1h = np.sqrt(1./Nin)
    w1l = b1l = -w1h

    w2h = np.sqrt(1./N)
    w2l = -w2h

    uden = lambda low,high: 1./(high-low)

    w2norm, _ =  quad(lambda w: uden(w2l,w2h)*w**2, w2l,w2h)

    if nl=='sigmoid':
        nl, nl_p = sigmoid, sigmoid_p
    elif nl=='tanh':
        nl, nl_p = tanh, tanh_p
     #in case nl is string, but unknown, raise error 
    elif type(nl)==str: 
        raise ValueError

    ntk_w2 = np.zeros([2,2])
    ntk_b1 = np.zeros([2,2])
    ntk_w1 = np.zeros([2,2])
    for ii, y1 in enumerate([P0,Pplus]):
        for jj, y2 in enumerate([P0,Pplus]):
            
            this_int, err = dblquad(lambda w,b:  nl(w*y1+b)*nl(w*y2+b), w1l,w1h,b1l,b1h)
            ntk_w2[ii,jj] = N * uden(w1l,w1h) * uden(b1l,b1h) * this_int
            
            this_int, err = dblquad(lambda w,b:  nl_p(w*y1+b)*nl_p(w*y2+b), w1l,w1h,b1l,b1h)
            ntk_b1[ii,jj] = N * w2norm * uden(w1l,w1h) * uden(b1l,b1h) * this_int
            ntk_w1[ii,jj] = N * w2norm * uden(w1l,w1h) * uden(b1l,b1h) * this_int *y1*y2

    ntk = ntk_w1 + ntk_b1 + ntk_w2

    if return_components:
        return ntk, ntk_w1, ntk_b1, ntk_w2
    else:
        return ntk
    
def compute_dfdx_tensor_np(model, x_vec, device="cpu"):
    """
    Compute the gradient of model outputs w.r.t. inputs:
        dfdx[i, :] = nabla_x f(x_vec[i])
    Returns an array of shape (N, D).
    """
    model.to(device)
    if not torch.is_tensor(x_vec):
        x_vec = torch.tensor(x_vec, dtype=torch.float32)
    x_vec = x_vec.to(device)
    N, D = x_vec.shape if x_vec.ndim == 2 else (1, x_vec.shape[0])
    dfdx = np.zeros((N, D), dtype=np.float64)
    # Assume scalar output model
    for i in range(N):
        # enable grad on input
        xi = x_vec[i].clone().detach().requires_grad_(True)
        model.zero_grad()
        yi = model(xi)
        # compute gradient of yi w.r.t. xi
        grad_x = torch.autograd.grad(yi, xi)[0]
        dfdx[i, :] = grad_x.detach().cpu().numpy()
    return dfdx

if __name__ == "__main__":
    '''
    validation of torch autograd against finite differences
    '''
    # Use double precision for better numeric stability
    dtype = torch.float64
    torch.manual_seed(0)

    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        def forward(self, x):
            return self.net(x)

    N, D = 5, 3
    x = torch.randn(N, D, dtype=dtype)
    print(f'x requires grad: {x.requires_grad}')
    model = MLP(D, 1024, 1).double()

    epsilon = 1e-1
    # Compute baseline NTK on unperturbed data
    ntk0 = compute_ntk_matrix_np(model, x)

    # 2. Build perturbed dataset: x_p of shape (N*D, D)
    xp_list = []
    for j in range(N):
        for k in range(D):
            x_pert = x.clone()
            x_pert[j, k] += epsilon
            xp_list.append(x_pert[j])
    xp = torch.stack(xp_list)
    print("==========================================================Testing dntk_dxj computation...")

    print(f"Perturbed dataset shape: {xp.shape} (should be {N*D}, {D})")
    # 3. Concatenate original and perturbed, compute full NTK
    big_X = torch.cat([x, xp], dim=0)
    big_ntk = compute_ntk_matrix_np(model, big_X)

    # 4. Extract off-diagonal block, reshape to (N, N, D), then take difference
    block = big_ntk[:N, N:].reshape(N, N, D)
    # Subtract unperturbed NTK for each (i,j) along D
    approx_dntk = (block - ntk0[:, :, None]) / epsilon

    # 4. Compute analytic dNTK via the provided function
    analytic_dntk = compute_dntk_dxj_tensor_np(model, x)



    # 5. Report accuracy metrics
    diff = np.abs(approx_dntk - analytic_dntk)
    max_error = diff.max()
    mean_error = diff.mean()
    rel_error = mean_error / (np.abs(analytic_dntk).mean() + 1e-12)

    print("Approximate dNTK vs Analytic dNTK (excerpt):")
    print(f"approx_dntk[:2, :2, :]:\n {approx_dntk[:2, :2, :]}\n"
          f"analytic_dntk[:2, :2, :]:\n {analytic_dntk[:2, :2, :]}")
    print(f"Max absolute error: {max_error:.3e}")
    print(f"Mean absolute error: {mean_error:.3e}")
    print(f"Relative mean error: {rel_error:.3e}")

    print("==========================================================Testing dfdx computation...")
    analytic_dfdx = compute_dfdx_tensor_np(model, x)
    fx = model(x).detach().cpu().numpy()
    fx_p = model(xp).detach().cpu().numpy()
    print(f"Unperturbed model input shape: {x.shape}, output shape: {fx.shape}")
    print(f"Perturbed model input shape: {xp.shape}, output shape: {fx_p.shape}")
    approx_dfdx = (np.reshape(fx_p,[N,D]) - fx) / epsilon
    dfdx_diff = np.abs(approx_dfdx - analytic_dfdx)
    dfdx_max_error = dfdx_diff.max()
    dfdx_mean_error = dfdx_diff.mean()
    dfdx_rel_error = dfdx_mean_error / (np.abs(analytic_dfdx).mean() + 1e-12)
    print("Approximate dfdx vs Analytic dfdx (excerpt):")
    print(f"approx_dfdx[:2, :]:\n {approx_dfdx[:2, :]}\n"
          f"analytic_dfdx[:2, :]:\n {analytic_dfdx[:2, :]}")
    print(f"Max absolute error: {dfdx_max_error:.3e}")
    print(f"Mean absolute error: {dfdx_mean_error:.3e}")
    print(f"Relative mean error: {dfdx_rel_error:.3e}") 