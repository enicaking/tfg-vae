import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import custom_bwd, custom_fwd

class DifferentiableClamp(torch.autograd.Function):
    """
    https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/6
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, min, max):
    """
    https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/6
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)


# Utility function for forward function
def sample_from_qz_given_x(qi, beta=torch.tensor(10), n_samples=1):

    """
    This method implements the DVAE's reparameterization trick.

    Parameters
        ----------
        qi : torch.tensor
            Probability of bits being 1.
        beta : torch.tensor
            Temperature term that controls the decay of the exponentials in the smoothing transformation. Default to 10.
        n_samples: int, optional
            Number of samples used to estimate the ELBO. Default to 1.

        Returns
        -------
        q_z: torch.tensor
            Sampled z.
    """

    # Here we are implementing the reparameterization trick from the DiscreteVAE

    # Sanity check
    assert torch.any(qi < 0)==False, "Negative value encountered in bit probabilities."
    assert torch.any(qi > 1)==False, "Value larger than 1 encountered in bit probabilities."


    # Obtain n_samples from q(z|x) {REPARAMETERIZATION}
    epsilon = 1e-6

    # Bit probabilities q(c_i=1|x)
    qi = qi.unsqueeze(2).repeat(1, 1, n_samples)
    ones = torch.ones((qi.shape)).to(qi.device)

    # Clamp to avoid divisions by 0 in the reparameterization
    qi = dclamp(qi, 0, 1-1e-3)

    # Sample from U(0,1)
    rho = torch.rand(qi.shape).to(qi.device)

    # Reparameterization
    b = (rho+torch.exp(-beta)*(qi-rho))/(ones-qi) - ones
    c = -(qi*torch.exp(-beta))/(ones-qi)

    dif = torch.sqrt(torch.pow(b, 2) - 4*c) - b
    dif = torch.where(dif <= 0, epsilon, dif)   # avoid negative or zero values due to numerical imprecission

    q_z = (-1/beta)*torch.log(dif/2) # shape [N, K, n_samples]

    # Sanity check
    assert torch.any(torch.isinf(q_z))==False, "Invalid q(z|x) value (inf)."
    assert torch.any(torch.isnan(q_z))==False, "Invalid q(z|x) value (nan)."

    return q_z


# Test utility function
def add_numbers(a, b):
    """
    This method adds two numbers.

    Parameters
        ----------
        a : int
            First number.
        b : int
            Second number.

        Returns
        -------
        sum: int
            Sum of a and b.
    """
    return a + b