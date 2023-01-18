"""
Definition of Canonical Link based on Convex Potential Flows
Modified from:
https://github.com/CW-Huang/CP-Flow/blob/main/lib/flows/cpflows.py
"""
import torch
import torch.nn.functional as F
import torch.optim

from lib.functional import log_softmax_plus
from lib.icnn import ICNN3


class DeepConvexFlow(torch.nn.Module):
    """
    Deep convex potential flow parameterized by an input-convex neural network.
    This is the main framework used in Huang et al. We use the forward pass of
    this model as v^{-1} within the (u,v)-geometric structure.
    """

    def __init__(
        self, icnn, bias_w1=0.0, trainable_mat=True, identity_diag=True
    ):
        super(DeepConvexFlow, self).__init__()
        self.icnn = icnn

        self.w1 = torch.nn.Parameter(torch.randn(1) + bias_w1)
        self.identity_diag = identity_diag
        if identity_diag:
            self.w0 = torch.nn.Parameter(
                torch.randn(1), requires_grad=trainable_mat
            )
        else:
            self.diags = torch.nn.Parameter(
                torch.log(torch.exp(torch.ones(icnn.dim, 1)) - 1),
                requires_grad=trainable_mat,
            )

        self.bias_w1 = bias_w1

    def get_potential(self, x, context=None):
        n = x.size(0)
        if context is None:
            icnn = self.icnn(x)
        else:
            icnn = self.icnn(x, context)

        if self.identity_diag:
            return (
                F.softplus(self.w1) * icnn
                + F.softplus(self.w0)
                * (x.view(n, -1) ** 2).sum(1, keepdim=True)
                / 2
            )
        else:
            return (
                F.softplus(self.w1) * icnn
                + torch.matmul(x.view(n, -1) ** 2, F.softplus(self.diags)) / 2
            )

    def forward(self, x, context=None):
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            F = self.get_potential(x, context)
            f = torch.autograd.grad(F.sum(), x, create_graph=True)[0]
        return f


class LegendreLink(torch.nn.Module):
    def __init__(
        self,
        n_blocks,
        K,
        dim_hidden=4,
        depth=4,
        trainable_mat=True,
        symm_act_first=False,
        softplus_type="softplus",
        zero_softplus=False,
        bias_w1=0.0,
        identity_diag=True,
    ):
        super(LegendreLink, self).__init__()

        cpflows = list()
        for _ in range(n_blocks):
            icnn = ICNN3(
                dim=K,
                dimh=dim_hidden,
                num_hidden_layers=depth,
                symm_act_first=symm_act_first,
                softplus_type=softplus_type,
                zero_softplus=zero_softplus,
            )
            cpflows.append(
                DeepConvexFlow(
                    icnn=icnn,
                    bias_w1=bias_w1,
                    trainable_mat=trainable_mat,
                    identity_diag=identity_diag,
                )
            )
        self.cpflows = torch.nn.ModuleList(cpflows)

        self.gpu_maybe = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        for cpflow in self.cpflows:
            cpflow.to(self.gpu_maybe)

    def forward(self, x, context=None):
        z = x
        for cpflow in self.cpflows:
            z = cpflow(z, context)
        logp_tilde = log_softmax_plus(z)

        return logp_tilde


class MultinomialLogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultinomialLogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.gpu_maybe = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.linear.to(self.gpu_maybe)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs
