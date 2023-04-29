# for comparision - pytorch pre-implemented optimizers
from torch.optim import SGD, RMSprop, Adam

# base class for subclassing
from torch.optim import Optimizer

import torch
from torch.nn.parameter import Parameter
from typing import Optional

# SGD, RMSprop, Adam implmentation from scratch
class _SGD_(Optimizer):
    def __init__(
        self,
        params,
        lr,
        momentum=0,
        weight_decay=0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def __init_state__(self, state, p):
        state["momentum_buffer"] = torch.zeros_like(p)

    def step(self):
        lr = self.defaults["lr"]
        momentum = self.defaults["momentum"]
        weight_decay = self.defaults["weight_decay"]

        # Loop through each parameter in each parameter group
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    # Get momentum buffer
                    state = self.state[p]
                    if len(state) == 0:
                        self.__init_state__(state, p)
                    momentum_buffer = state["momentum_buffer"]

                    # Get gradient of param
                    if p.grad is None:
                        continue
                    grad = p.grad.data 

                    # Add weight decay
                    if weight_decay != 0:
                        grad += weight_decay * p

                    # add momentum to gradient
                    buffer = momentum * momentum_buffer + grad

                    # update parameter
                    p -= lr * buffer

                    # update momentum buffer
                    state["momentum_buffer"] = buffer 


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
        )
        super().__init__(params, defaults)

    def __init_state__(self, state, p):
        state["sq_avg"] = torch.zeros_like(p)
        state["buffer"] = torch.zeros_like(p)
        state["avg_grad"] = torch.zeros_like(p)

    def step(self, closure=None):
        centered = self.defaults["centered"]
        lr = self.defaults["lr"]
        alpha = self.defaults["alpha"]
        momentum = self.defaults["momentum"]
        eps = self.defaults["eps"]
        weight_decay = self.defaults["weight_decay"]

        # Loop through each parameter in each parameter group
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
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
                    state["sq_avg"] = rms
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
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        )
        super().__init__(params, defaults)

    def __init_state__(self, state, p):
        state["first_moment"] = torch.zeros_like(p)
        state["second_moment"] = torch.zeros_like(p)
        state["max_second_moment"] = torch.zeros_like(p)

    def step(self, closure=None):
        lr = self.defaults["lr"]
        eps = self.defaults["eps"]
        beta_1, beta_2 = self.defaults["betas"]
        weight_decay = self.defaults["weight_decay"]

        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    # Get "state" of parameter (stores RMS, momentum, etc)
                    state = self.state[p]
                    if len(state) == 0:
                        self.__init_state__(state, p)
                    first_moment = state["first_moment"]
                    second_moment = state["second_moment"]
                    max_second_moment = state["max_second_moment"]

                    # Get gradient of param
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if weight_decay != 0:
                        grad += weight_decay * p

                    # Update moments
                    first_moment = beta_1 * first_moment + (1-beta_1) * grad
                    second_moment = beta_2 * second_moment + (1-beta_2) * grad ** 2
                    state["first_moment"] = first_moment
                    state["second_moment"] = second_moment

                    normalized_first_moment = first_moment / (1 - beta_1)
                    normalized_second_moment = second_moment / (1 - beta_2)

                    # Apply computed update
                    if self.defaults["amsgrad"]:
                        max_second_moment = max(max_second_moment, normalized_second_moment)
                        state["max_second_moment"] = max_second_moment
                        p -= lr * normalized_first_moment / (max_second_moment.sqrt() + eps)
                    else:
                        p -= lr * normalized_first_moment / (normalized_second_moment.sqrt() + eps)
