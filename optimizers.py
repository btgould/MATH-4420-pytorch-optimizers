# for comparision - pytorch pre-implemented optimizers
from torch.optim import SGD, RMSprop, Adam

# base class for subclassing
from torch.optim import Optimizer

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
        raise NotImplementedError

    def step(self, closure=None):
        raise NotImplementedError


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
