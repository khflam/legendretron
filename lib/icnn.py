"""
Definitions of Input Convex Neural Networks and related helper functions
Taken from:
https://github.com/CW-Huang/CP-Flow/blob/main/lib/icnn.py
"""
import numpy as np
import torch
from torch import Tensor, nn

_scaling_min = 0.001


def symm_softplus(x, softplus_=torch.nn.functional.softplus):
    return softplus_(x) - 0.5 * x


def softplus(x):
    return nn.functional.softplus(x)


def gaussian_softplus(x):
    z = np.sqrt(np.pi / 2)
    return (
        z * x * torch.erf(x / np.sqrt(2)) + torch.exp(-(x**2) / 2) + z * x
    ) / (2 * z)


def gaussian_softplus2(x):
    z = np.sqrt(np.pi / 2)
    return (
        z * x * torch.erf(x / np.sqrt(2)) + torch.exp(-(x**2) / 2) + z * x
    ) / z


def laplace_softplus(x):
    return torch.relu(x) + torch.exp(-torch.abs(x)) / 2


def cauchy_softplus(x):
    # (Pi y + 2 y ArcTan[y] - Log[1 + y ^ 2]) / (2 Pi)
    pi = np.pi
    return (x * pi - torch.log(x**2 + 1) + 2 * x * torch.atan(x)) / (2 * pi)


def activation_shifting(activation):
    def shifted_activation(x):
        return activation(x) - activation(torch.zeros_like(x))

    return shifted_activation


def get_softplus(softplus_type="softplus", zero_softplus=False):
    if softplus_type == "softplus":
        act = nn.functional.softplus
    elif softplus_type == "gaussian_softplus":
        act = gaussian_softplus
    elif softplus_type == "gaussian_softplus2":
        act = gaussian_softplus2
    elif softplus_type == "laplace_softplus":
        act = laplace_softplus
    elif softplus_type == "cauchy_softplus":
        act = cauchy_softplus
    else:
        raise NotImplementedError(
            f"softplus type {softplus_type} not supported."
        )
    if zero_softplus:
        act = activation_shifting(act)
    return act


class Softplus(nn.Module):
    def __init__(self, softplus_type="softplus", zero_softplus=False):
        super(Softplus, self).__init__()
        self.softplus_type = softplus_type
        self.zero_softplus = zero_softplus

    def forward(self, x):
        return get_softplus(self.softplus_type, self.zero_softplus)(x)


class SymmSoftplus(torch.nn.Module):
    def forward(self, x):
        return symm_softplus(x)


class PosLinear(torch.nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        gain = 1 / x.size(1)
        return (
            nn.functional.linear(
                x, torch.nn.functional.softplus(self.weight), self.bias
            )
            * gain
        )


class ActNorm(torch.nn.Module):
    """ActNorm layer with data-dependant init."""

    def __init__(
        self, num_features, logscale_factor=1.0, scale=1.0, learn_scale=True
    ):
        super(ActNorm, self).__init__()
        self.initialized = False
        self.num_features = num_features

        self.register_parameter(
            "b",
            nn.Parameter(torch.zeros(1, num_features, 1), requires_grad=True),
        )
        self.learn_scale = learn_scale
        if learn_scale:
            self.logscale_factor = logscale_factor
            self.scale = scale
            self.register_parameter(
                "logs",
                nn.Parameter(
                    torch.zeros(1, num_features, 1), requires_grad=True
                ),
            )

    def forward_transform(self, x, logdet=0):
        input_shape = x.size()
        x = x.view(input_shape[0], input_shape[1], -1)

        if not self.initialized:
            self.initialized = True

            # noinspection PyShadowingNames
            def unsqueeze(x):
                return x.unsqueeze(0).unsqueeze(-1).detach()

            # Compute the mean and variance
            sum_size = x.size(0) * x.size(-1)
            b = -torch.sum(x, dim=(0, -1)) / sum_size
            self.b.data.copy_(unsqueeze(b).data)

            if self.learn_scale:
                var = unsqueeze(
                    torch.sum((x + unsqueeze(b)) ** 2, dim=(0, -1)) / sum_size
                )
                logs = (
                    torch.log(self.scale / (torch.sqrt(var) + 1e-6))
                    / self.logscale_factor
                )
                self.logs.data.copy_(logs.data)

        b = self.b
        output = x + b

        if self.learn_scale:
            logs = self.logs * self.logscale_factor
            scale = torch.exp(logs) + _scaling_min
            output = output * scale
            dlogdet = torch.sum(torch.log(scale)) * x.size(-1)  # c x h

            return output.view(input_shape), logdet + dlogdet
        else:
            return output.view(input_shape), logdet

    def reverse(self, y, **kwargs):
        assert self.initialized
        input_shape = y.size()
        y = y.view(input_shape[0], input_shape[1], -1)
        logs = self.logs * self.logscale_factor
        b = self.b
        scale = torch.exp(logs) + _scaling_min
        x = y / scale - b

        return x.view(input_shape)

    def extra_repr(self):
        return f"{self.num_features}"


class ActNormNoLogdet(ActNorm):
    def forward(self, x):
        return super(ActNormNoLogdet, self).forward_transform(x)[0]


class ICNN3(torch.nn.Module):
    def __init__(
        self,
        dim=2,
        dimh=16,
        num_hidden_layers=2,
        symm_act_first=False,
        softplus_type="softplus",
        zero_softplus=False,
    ):
        super(ICNN3, self).__init__()
        self.dim = dim

        self.act = Softplus(
            softplus_type=softplus_type, zero_softplus=zero_softplus
        )
        self.symm_act_first = symm_act_first

        Wzs = list()
        Wzs.append(nn.Linear(dim, dimh))
        for _ in range(num_hidden_layers - 1):
            Wzs.append(PosLinear(dimh, dimh // 2, bias=True))
        Wzs.append(PosLinear(dimh, 1, bias=False))
        self.Wzs = torch.nn.ModuleList(Wzs)

        Wxs = list()
        for _ in range(num_hidden_layers - 1):
            Wxs.append(nn.Linear(dim, dimh // 2))
        Wxs.append(nn.Linear(dim, 1, bias=False))
        self.Wxs = torch.nn.ModuleList(Wxs)

        Wx2s = list()
        for _ in range(num_hidden_layers - 1):
            Wx2s.append(nn.Linear(dim, dimh // 2))
        self.Wx2s = torch.nn.ModuleList(Wx2s)

        actnorms = list()
        for _ in range(num_hidden_layers - 1):
            actnorms.append(ActNormNoLogdet(dimh // 2))
        actnorms.append(ActNormNoLogdet(1))
        actnorms[-1].b.requires_grad_(False)
        self.actnorms = torch.nn.ModuleList(actnorms)

    def forward(self, x):
        if self.symm_act_first:
            z = symm_softplus(self.Wzs[0](x), self.act)
        else:
            z = self.act(self.Wzs[0](x))
        for Wz, Wx, Wx2, actnorm in zip(
            self.Wzs[1:-1], self.Wxs[:-1], self.Wx2s[:], self.actnorms[:-1]
        ):
            z = self.act(actnorm(Wz(z) + Wx(x)))
            aug = Wx2(x)
            aug = (
                symm_softplus(aug, self.act)
                if self.symm_act_first
                else self.act(aug)
            )
            z = torch.cat([z, aug], 1)
        return self.actnorms[-1](self.Wzs[-1](z) + self.Wxs[-1](x))
