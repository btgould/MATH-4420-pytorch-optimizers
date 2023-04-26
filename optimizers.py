# for comparision - pytorch pre-implemented optimizers
from torch.optim import SGD, RMSprop, Adam

# base class for subclassing
from torch.optim import Optimizer

import torch
from torch.nn.parameter import Parameter
from typing import Optional

# implmentation from scratch
class _SGD_(Optimizer):
    def __init__(
        self,
        params,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False
    ):
        raise NotImplementedError

    def step(self, closure=None):
        raise NotImplementedError


class _RMSprop_(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-2,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0,
        momentum=0,
        centered=False,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        differentiable: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            alpha=alpha,
            eps=eps,
            centered=centered,
            weight_decay=weight_decay,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
        )
        super().__init__(params, defaults)

    def __init_state__(self, state, p):
        state["sq_avg"] = torch.zeros_like(p)
        state["buffer"] = torch.zeros_like(p)
        state["avg_grad"] = torch.zeros_like(p)

    def step(self, closure=None):
        centered = self.defaults["centered"]

        # Loop through each parameter in each parameter group
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    lr = self.defaults["lr"]
                    alpha = self.defaults["alpha"]
                    momentum = self.defaults["momentum"]
                    eps = self.defaults["eps"]
                    weight_decay = self.defaults["weight_decay"]

                    # Get "state" of parameter (stores RMS, momentum, etc)
                    state = self.state[p]
                    if len(state) == 0:
                        self.__init_state__(state, p)
                    rms = state["sq_avg"]
                    avg_grad = state["avg_grad"]
                    buffer = state["buffer"]

                    # Get gradient of param
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if weight_decay != 0:
                        grad += weight_decay * p

                    # Get RMS of param's gradient
                    rms = alpha * rms + (1 - alpha) * grad * grad
                    centered_rms = rms

                    # Centering: instead of normalizing by magnitude of the grad,
                    # normalize by magnitude of difference between current grad and average grad
                    if centered:
                        avg_grad = alpha * avg_grad + (1 - alpha) * grad
                        state["avg_grad"] = avg_grad
                        centered_rms -= avg_grad**2

                    normalized_grad = grad / (centered_rms.sqrt() + eps)

                    # Apply computed update 
                    if momentum > 0:
                        # Use update with accumulated momentum
                        buffer = momentum * buffer + normalized_grad
                        p -= lr * buffer
                        state["buffer"] = buffer 
                    else:
                        # Use update on own 
                        p -= lr * normalized_grad


class _Adam_(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None
    ):
        raise NotImplementedError

    def step(self, closure=None):
        raise NotImplementedError
