import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math

# https://github.com/allenai/hidden-networks
class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class GetSubnetNM(autograd.Function):
    @staticmethod
    def forward(ctx, scores, keep_n, block_m):
        # N:M pruning: split weights into contiguous blocks of size M and keep
        # the top-N scores within each block (per output channel/neuron).
        if keep_n > block_m:
            raise ValueError("N:M pruning requires n <= m.")
        if keep_n <= 0 or block_m <= 0:
            raise ValueError("N:M pruning requires n > 0 and m > 0.")
        if scores.dim() < 2:
            raise ValueError("N:M pruning expects scores with at least 2 dimensions.")

        original_shape = scores.shape
        scores_flat = scores.reshape(scores.shape[0], -1)
        original_len = scores_flat.size(1)
        pad = (block_m - (scores_flat.size(1) % block_m)) % block_m
        if pad:
            scores_flat = F.pad(scores_flat, (0, pad), value=float("-inf"))
        scores_grouped = scores_flat.view(scores_flat.size(0), -1, block_m)

        topk_indices = scores_grouped.topk(keep_n, dim=-1).indices
        mask_grouped = torch.zeros_like(scores_grouped)
        mask_grouped.scatter_(-1, topk_indices, 1.0)

        mask_flat = mask_grouped.view(scores_flat.size(0), -1)
        if pad:
            mask_flat = mask_flat[:, :original_len]
        return mask_flat.view(original_shape)

    @staticmethod
    def backward(ctx, g):
        return g, None, None


def build_pruning_mask(scores, k, prune_type, structured_dim=0, nm_n=None, nm_m=None):
    abs_scores = scores.abs()
    if prune_type == "unstructured":
        # Unstructured: identical to original GetSubnet over all weights.
        return GetSubnet.apply(abs_scores, k)
    if prune_type == "structured":
        # Structured: prune per output channel/neuron (dim=0 only).
        if structured_dim != 0:
            raise ValueError("Structured pruning only supports dim=0.")
        if abs_scores.dim() == 4:
            # Conv2d: aggregate scores over (in_channels, kH, kW).
            channel_scores = abs_scores.mean(dim=(1, 2, 3))
            channel_mask = GetSubnet.apply(channel_scores, k)
            return channel_mask[:, None, None, None].expand_as(abs_scores)
        if abs_scores.dim() == 2:
            # Linear: aggregate scores over input features.
            channel_scores = abs_scores.mean(dim=1)
            channel_mask = GetSubnet.apply(channel_scores, k)
            return channel_mask[:, None].expand_as(abs_scores)
        raise ValueError("Structured pruning only supports Conv2d and Linear layers.")
    if prune_type == "nm":
        # N:M: keep nm_n weights per block of size nm_m within each row.
        # For Conv2d, follow Apex-style layout by permuting to (kH, kW, out, in)
        # before grouping into blocks, then permute back.
        if nm_n is None or nm_m is None:
            raise ValueError("N:M pruning requires nm_n and nm_m settings.")
        if abs_scores.dim() == 4:
            scores_nm = abs_scores.permute(2, 3, 0, 1)
            mask_nm = GetSubnetNM.apply(scores_nm, nm_n, nm_m)
            return mask_nm.permute(2, 3, 0, 1)
        return GetSubnetNM.apply(abs_scores, nm_n, nm_m)
    raise ValueError(f"Unknown prune_type '{prune_type}'.")


class SubnetConv(nn.Conv2d):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight
    # Gradients to self.weight, self.bias have been turned off by default.

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(SubnetConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.popup_scores = Parameter(torch.Tensor(self.weight.shape))
        nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.w = 0
        self.prune_type = "unstructured"
        self.structured_dim = 0
        self.nm_n = None
        self.nm_m = None

    def set_prune_rate(self, k):
        self.k = k

    def set_prune_config(self, prune_type, structured_dim=0, nm_n=None, nm_m=None):
        self.prune_type = prune_type
        self.structured_dim = structured_dim
        self.nm_n = nm_n
        self.nm_m = nm_m

    def forward(self, x):
        # Get the subnetwork by sorting the scores.
        adj = build_pruning_mask(
            self.popup_scores,
            self.k,
            self.prune_type,
            structured_dim=self.structured_dim,
            nm_n=self.nm_n,
            nm_m=self.nm_m,
        )

        # Use only the subnetwork in the forward pass.
        self.w = self.weight * adj
        x = F.conv2d(
            x, self.w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class SubnetLinear(nn.Linear):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight
    # Gradients to self.weight, self.bias have been turned off.

    def __init__(self, in_features, out_features, bias=True):
        super(SubnetLinear, self).__init__(in_features, out_features, bias=True)
        self.popup_scores = Parameter(torch.Tensor(self.weight.shape))
        nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.w = 0
        # self.register_buffer('w', None)
        self.prune_type = "unstructured"
        self.structured_dim = 0
        self.nm_n = None
        self.nm_m = None

    def set_prune_rate(self, k):
        self.k = k

    def set_prune_config(self, prune_type, structured_dim=0, nm_n=None, nm_m=None):
        self.prune_type = prune_type
        self.structured_dim = structured_dim
        self.nm_n = nm_n
        self.nm_m = nm_m

    def forward(self, x):
        # Get the subnetwork by sorting the scores.
        adj = build_pruning_mask(
            self.popup_scores,
            self.k,
            self.prune_type,
            structured_dim=self.structured_dim,
            nm_n=self.nm_n,
            nm_m=self.nm_m,
        )

        # Use only the subnetwork in the forward pass.
        self.w = self.weight * adj
        x = F.linear(x, self.w, self.bias)

        return x
