"""Argmin-differentiable total variation functions."""
from __future__ import division, print_function

# Must import these first, gomp issues with pytorch.
from prox_tv import tv1w_2d, tv1_2d, tv1w_1d, tv1_1d

import numpy as np
from sklearn.isotonic import isotonic_regression as isotonic

import torch
from torch.autograd import Function
from .blocks import blockwise_means, blocks_2d

__all__ = ("TotalVariation2d", "TotalVariation2dWeighted", "TotalVariation1d")


class TotalVariationBase(Function):

    @staticmethod
    def _grad_x(opt, grad_output, average_connected):
        if average_connected:
            blocks = blocks_2d(opt.detach().cpu().numpy())
        else:
            _, blocks = np.unique(opt.detach().cpu().numpy().ravel(),
                                  return_inverse=True)
        grad_x = blockwise_means(blocks.ravel(),
                                 grad_output.detach().cpu().numpy().ravel())
        # We need the clone as there seems to e a double-free error in py27,
        # namely, torch free()s the array after numpy has already free()d it.
        return torch.from_numpy(grad_x).view(opt.size()).clone()

    @staticmethod
    def _grad_w_row(opt, grad_x):
        """Compute the derivative with respect to the row weights."""
        diffs_row = torch.sign(opt[:, :-1] - opt[:, 1:])
        return - diffs_row * (grad_x[:, :-1] - grad_x[:, 1:])

    @staticmethod
    def _grad_w_col(opt, grad_x):
        """Compute the derivative with respect to the column weights."""
        diffs_col = torch.sign(opt[:-1, :] - opt[1:, :])
        return - diffs_col * (grad_x[:-1, :] - grad_x[1:, :])

    @staticmethod
    def _refine(opt, x, weights_row, weights_col):
        """Refine the solution by solving an isotonic regression.

        The weights can either be two-dimensional tensors, or of shape (1,)."""
        idx = np.argsort(opt.ravel())  # Will pick an arbitrary order cone.
        ordered_vec = np.zeros_like(idx, dtype=np.float)
        ordered_vec[idx] = np.arange(np.size(opt))
        f = TotalVariationBase._linearize(ordered_vec.reshape(opt.shape),
                                          weights_row.detach().cpu().numpy(),
                                          weights_col.detach().cpu().numpy())
        opt_idx = isotonic((x.view(-1).detach().cpu().numpy() - f.ravel())[idx])
        opt = np.zeros_like(opt_idx)
        opt[idx] = opt_idx
        return opt

    @staticmethod
    def _linearize(y, weights_row, weights_col):
        """Compute a linearization of the graph-cut function at the given point.

        Arguments
        ---------
        y : numpy.ndarray
            The point where the linearization is computed, shape ``(m, n)``.
        weights_row : numpy.ndarray
            The non-negative row weights, with shape ``(m, n - 1)``.
        y : numpy.ndarray
            The non-negative column weights, with shape ``(m - 1, n)``.

        Returns
        -------
        numpy.ndarray
            The linearization of the graph-cut function at ``y``."""
        diffs_col = np.sign(y[1:, :] - y[:-1, :])
        diffs_row = np.sign(y[:, 1:] - y[:, :-1])

        f = np.zeros_like(y)  # The linearization.
        f[:, 1:] += diffs_row * weights_row
        f[:, :-1] -= diffs_row * weights_row
        f[1:, :] += diffs_col * weights_col
        f[:-1, :] -= diffs_col * weights_col

        return f


def TotalVariation2dWeighted(refine=True, average_connected=True, tv_args={}):
    r"""A two dimensional total variation function.

    Specifically, given as input the unaries `x`, positive row weights
    :math:`\mathbf{r}` and positive column weights :math:`\mathbf{c}`, the
    output is computed as

    .. math::

        \textrm{argmin}_{\mathbf z}
            \frac{1}{2} \|\mathbf{x}-\mathbf{z}\|^2 +
            \sum_{i, j} r_{i,j} |z_{i, j} - z_{i, j + 1}| +
            \sum_{i, j} c_{i,j} |z_{i, j} - z_{i + 1, j}|.

    Arguments
    ---------
        refine: bool
            If set the solution will be refined with isotonic regression.
        average_connected: bool
            How to compute the approximate derivative.

            If ``True``, will average within each connected component.
            If ``False``, it will average within each block of equal values.
            Typically, you want this set to true.
        tv_args: dict
            The dictionary of arguments passed to the total variation solver.
        """

    class TotalVariation2dWeighted_(TotalVariationBase):

        @staticmethod
        def forward(ctx, x, weights_row, weights_col):
            r"""Solve the total variation problem and return the solution.

            Arguments
            ---------
                x: :class:`torch:torch.Tensor`
                    A tensor with shape ``(m, n)`` holding the input signal.
                weights_row: :class:`torch:torch.Tensor`
                    The horizontal edge weights.

                    Tensor of shape ``(m, n - 1)``, or ``(1,)`` if all weights
                    are equal.
                weights_col: :class:`torch:torch.Tensor`
                    The vertical edge weights.

                    Tensor of shape ``(m - 1, n)``, or ``(1,)`` if all weights
                    are equal.

            Returns
            -------
            :class:`torch:torch.Tensor`
                The solution to the total variation problem, of shape ``(m, n)``.
            """
            opt = tv1w_2d(x.detach().cpu().numpy(),
                          weights_col.detach().cpu().numpy(),
                          weights_row.detach().cpu().numpy(),
                          **tv_args)
            if refine:
                opt = TotalVariation2dWeighted_._refine(
                    opt, x, weights_row, weights_col)
            opt = torch.tensor(opt, device=x.device).view_as(x)
            ctx.save_for_backward(opt)
            ctx.device = x.device
            return opt

        @staticmethod
        def backward(ctx, grad_output):
            opt, = ctx.saved_tensors
            grad_weights_row, grad_weights_col = None, None
            grad_x = TotalVariation2dWeighted_._grad_x(
                opt, grad_output, average_connected).to(ctx.device)

            if ctx.needs_input_grad[1]:
                grad_weights_row = TotalVariation2dWeighted_._grad_w_row(
                    opt, grad_x).to(ctx.device)

            if ctx.needs_input_grad[2]:
                grad_weights_col = TotalVariation2dWeighted_._grad_w_col(
                    opt, grad_x).to(ctx.device)

            return grad_x, grad_weights_row, grad_weights_col

    return TotalVariation2dWeighted_.apply


def TotalVariation2d(refine=True, average_connected=True, tv_args={}):
    r"""A two dimensional total variation function with tied edge weights.

    Specifically, given as input the unaries `x` and edge weight ``w``, the
    returned value is given by:

    .. math::

        \textrm{argmin}_{\mathbf z}
            \frac{1}{2} \|\mathbf{x}-\mathbf{z}\|^2 +
            \sum_{i, j} w |z_{i, j} - z_{i, j + 1}| +
            \sum_{i, j} w |z_{i, j} - z_{i + 1, j}|.

    Arguments
    ---------
        refine: bool
            If set the solution will be refined with isotonic regression.
        average_connected: bool
            How to compute the approximate derivative.

            If ``True``, will average within each connected component.
            If ``False``, it will average within each block of equal values.
            Typically, you want this set to true.
        tv_args: dict
            The dictionary of arguments passed to the total variation solver.
        """

    class TotalVariation2d_(TotalVariationBase):

        @staticmethod
        def forward(ctx, x, w):
            r"""Solve the total variation problem and return the solution.

            Arguments
            ---------
                x: :class:`torch:torch.Tensor`
                    A tensor with shape ``(m, n)`` holding the input signal.
                weights_row: :class:`torch:torch.Tensor`
                    The horizontal edge weights.

                    Tensor of shape ``(m, n - 1)``, or ``(1,)`` if all weights
                    are equal.
                weights_col: :class:`torch:torch.Tensor`
                    The vertical edge weights.

                    Tensor of shape ``(m - 1, n)``, or ``(1,)`` if all weights
                    are equal.

            Returns
            -------
            :class:`torch:torch.Tensor`
                The solution to the total variation problem, of shape ``(m, n)``.
            """
            assert w.size() == (1,)
            opt = tv1_2d(x.detach().cpu().numpy(),
                         w.detach().cpu().numpy()[0],
                         **tv_args)

            if refine:  # Should we improve it with isotonic regression.
                opt = TotalVariation2d_._refine(opt, x, w, w)

            opt = torch.tensor(opt, device=x.device).view_as(x)
            ctx.save_for_backward(opt)
            ctx.device = x.device
            return opt

        @staticmethod
        def backward(ctx, grad_output):
            opt, = ctx.saved_tensors
            grad_x = TotalVariation2d_._grad_x(
                opt, grad_output, average_connected).to(ctx.device)
            grad_w = None

            if ctx.needs_input_grad[1]:
                grad_w = (
                    torch.sum(TotalVariation2d_._grad_w_row(opt, grad_x)) +
                    torch.sum(TotalVariation2d_._grad_w_col(opt, grad_x))
                )
                grad_w = torch.tensor([grad_w], device=ctx.device)

            return grad_x, grad_w

    return TotalVariation2d_.apply


def TotalVariation1d(average_connected=True, tv_args={}):
    r"""A one dimensional total variation function.

    Specifically, given as input the signal `x` and weights :math:`\mathbf{w}`,
    the output is computed as

    .. math::

        \textrm{argmin}_{\mathbf z}
            \frac{1}{2} \|\mathbf{x}-\mathbf{z}\|^2 +
            \sum_{i=1}^{n-1} w_i |z_i - z_{i+1}|.

    Arguments
    ---------
        average_connected: bool
            How to compute the approximate derivative.

            If ``True``, will average within each connected component.
            If ``False``, it will average within each block of equal values.
            Typically, you want this set to true.
        tv_args: dict
            The dictionary of arguments passed to the total variation solver.
        """

    class TotalVariation1d_(TotalVariationBase):

        @staticmethod
        def forward(ctx, x, weights):
            r"""Solve the total variation problem and return the solution.

            Arguments
            ---------
                x: :class:`torch:torch.Tensor`
                    A tensor with shape ``(n,)`` holding the input signal.
                weights: :class:`torch:torch.Tensor`
                    The edge weights.

                    Shape ``(n-1,)``, or ``(1,)`` if all weights are equal.

            Returns
            -------
            :class:`torch:torch.Tensor`
                The solution to the total variation problem, of shape ``(m, n)``
            """
            ctx.equal_weights = weights.size() == (1,)
            if ctx.equal_weights:
                opt = tv1_1d(x.detach().cpu().numpy().ravel(),
                             weights.detach().cpu().numpy()[0],
                             **tv_args)
            else:
                opt = tv1w_1d(x.detach().cpu().numpy().ravel(),
                              weights.detach().cpu().numpy().ravel(),
                              **tv_args)
            opt = torch.tensor(opt, device=x.device).view_as(x)

            ctx.save_for_backward(opt)
            ctx.device = x.device
            return opt

        @staticmethod
        def backward(ctx, grad_output):
            opt, = ctx.saved_tensors
            grad_weights = None

            opt = opt.view((1, -1))
            grad_x = TotalVariation1d_._grad_x(
                opt, grad_output, average_connected).to(ctx.device)

            if ctx.needs_input_grad[1]:
                grad_weights = TotalVariation1d_._grad_w_row(
                    opt, grad_x).view(-1).to(ctx.device)
                if ctx.equal_weights:
                    grad_weights = torch.tensor([torch.sum(grad_weights)],
                                                device=ctx.device)

            return grad_x.view(-1), grad_weights

    return TotalVariation1d_.apply
