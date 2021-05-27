from __future__ import division, print_function
import torch
from torch.autograd import Variable
from torch_submod.graph_cuts import (
        TotalVariation2d, TotalVariation2dWeighted, TotalVariation1d)
from torch.autograd.gradcheck import gradcheck
from hypothesis import given, settings
import hypothesis.strategies as st


@settings(deadline=30000, max_examples=10)
@given(st.integers(2, 10), st.integers(10, 100), st.floats(0.1, 10))
def test_1d(b, n, w):
    x = Variable(torch.randn(b, n).reshape(b, n), requires_grad=True)
    w = Variable(torch.Tensor([[w]] * b), requires_grad=True)
    tv_args = {'method': 'condattautstring'}

    for batch in [True, False]:
        tv = TotalVariation1d(average_connected=False, num_workers=min(8, b),
                              multithread=True, batch_backward=batch, tv_args=tv_args)
        with torch.no_grad():
            batch_tv = tv(x, w)
            sample_tv = torch.stack([tv(x[i], w[i]) for i in range(b)])

        assert bool(torch.allclose(batch_tv, sample_tv))
        assert gradcheck(tv, (x, w), eps=1e-5, atol=1e-2, rtol=1e-3)


@settings(deadline=30000, max_examples=10)
@given(st.integers(2, 10), st.integers(10, 20), st.floats(0.1, 10))
def test_1dw(b, n, w):
    x = Variable(10 * torch.randn(b, n), requires_grad=True)
    w = Variable(0.1 + w * torch.rand(b, n - 1), requires_grad=True)
    tv_args = {'method': 'tautstring'}

    for batch in [True, False]:
        tv = TotalVariation1d(average_connected=False, num_workers=min(8, b),
                              multithread=True, batch_backward=batch, tv_args=tv_args)
        with torch.no_grad():
            batch_tv = tv(x, w)
            sample_tv = torch.stack([tv(x[i], w[i]) for i in range(b)])

        assert bool(torch.allclose(batch_tv, sample_tv))
        assert gradcheck(tv, (x, w), eps=5e-5, atol=5e-2, rtol=1e-2)


@settings(deadline=30000, max_examples=10)
@given(st.integers(2, 10), st.integers(2, 5), st.integers(2, 5), st.floats(0.1, 10))
def test_2d(b, n, m, w):
    x = Variable(torch.randn(b, n, m), requires_grad=True)
    w = Variable(0.1 + torch.Tensor([[w]] * b), requires_grad=True)
    tv_args = {'method': 'dr', 'max_iters': 100, 'n_threads': 2}

    for batch in [True, False]:
        tv = TotalVariation2d(refine=True, average_connected=False, num_workers=min(8, b),
                              multithread=True, batch_backward=batch, tv_args=tv_args)
        with torch.no_grad():
            batch_tv = tv(x, w)
            sample_tv = torch.stack([tv(x[i], w[i]) for i in range(b)])

        assert torch.allclose(batch_tv, sample_tv)
        assert gradcheck(tv, (x, w), eps=1e-5, atol=1e-2, rtol=1e-3)


@settings(deadline=30000, max_examples=10)
@given(st.integers(2, 10), st.integers(2, 5), st.integers(2, 5), st.floats(0.1, 10))
def test_2dw(b, n, m, w):
    x = Variable(torch.randn(b, n, m), requires_grad=True)
    w_r = Variable(0.1 + w * torch.rand(b, n, m-1), requires_grad=True)
    w_c = Variable(0.1 + w * torch.rand(b, n-1, m), requires_grad=True)
    tv_args = {'max_iters': 100, 'n_threads': 2}

    for batch in [True, False]:
        tv = TotalVariation2dWeighted(refine=True, average_connected=False,
                                      num_workers=min(8, b), multithread=True,
                                      batch_backward=batch, tv_args=tv_args)
        with torch.no_grad():
            batch_tv = tv(x, w_r, w_c)
            sample_tv = torch.stack([tv(x[i], w_r[i], w_c[i]) for i in range(b)])

        assert torch.allclose(batch_tv, sample_tv)
        assert gradcheck(tv, (x, w_r, w_c), eps=1e-5, atol=5e-2, rtol=1e-3)
