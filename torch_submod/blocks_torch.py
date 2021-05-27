import torch
import torch.nn.functional as F
from torch_scatter import scatter


def blockwise_means_batch(opt, inputs):
    if opt.ndim != 2 or inputs.ndim != 2:
        raise RuntimeError("You must provide a batch of 1-dimensional arrays. "
                           "Expected shape [batch, num_elems]")
    if opt.shape != inputs.shape:
        raise RuntimeError("'opt' and 'inputs' must have the same shape")

    B, N = opt.shape
    device = opt.device

    # Compute the range of value in each sample
    maxs = opt.reshape(B, -1).max(dim=-1).values  # [B]
    mins = opt.reshape(B, -1).min(dim=-1).values  # [B]
    mins[0] = 0  # Leave the first range of values where it is
    steps = torch.cumsum(maxs - mins + 1, dim=0)
    # Remove the last element and add a zero in first position
    steps = F.pad(steps[:-1], [1, 0]).reshape(-1, 1)

    # Offset the range of values in different samples in such a way that they
    # are consecutive and distant from each other by 1.
    # In the values axes we have:
    # ..., [sample0_min, ..., sample0_max], sample0_max+1,
    # [sample1_min, ..., sample1_max], sample1_max+1, ...
    # This ensures disjoint inverse indices returned by torch.unique
    opt_off = opt + steps

    # Compute the block indices (idx.shape == opt.shape)
    _, blocks = torch.unique(opt_off, return_inverse=True)

    # Compute the block averages
    means = scatter(inputs.reshape(-1).float(), blocks.reshape(-1), reduce="mean")
    # Copy the average value in all positions that were used to compute each mean
    output = means[blocks]

    return output.contiguous()
