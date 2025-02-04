import torch
import numpy as np
from scipy.integrate import quad, dblquad, tplquad

def compute_ntk_matrix_np(model, x_vec, device="cpu"):
    """Compute the NTK matrix for a given model and input vector, supporting CUDA if available."""
    model.to(device)
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