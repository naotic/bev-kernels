"""Patch nn.Conv2d.forward so plain 1x1 convs dispatch to a matmul (GEMM).
"""

import torch
import torch.nn as nn


class _Conv1x1MmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        n, cin, h, w_ = x.shape
        cout = weight.shape[0]
        wf = weight.detach().reshape(cout, cin)
        out = torch.matmul(wf, x.flatten(2)).view(n, cout, h, w_)
        if bias is not None:
            out = out + bias.reshape(1, -1, 1, 1)
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        ctx.shape = (n, cin, h, w_, cout)
        return out

    @staticmethod
    def backward(ctx, go):
        x, weight = ctx.saved_tensors
        n, cin, h, w_, cout = ctx.shape
        wf = weight.detach().reshape(cout, cin)
        go2 = go.reshape(n, cout, -1)
        grad_x = torch.matmul(wf.t(), go2).view(n, cin, h, w_)
        xf = x.detach().reshape(n, cin, -1)
        grad_w = torch.matmul(go2, xf.transpose(1, 2)).sum(0).view(cout, cin, 1, 1)
        grad_b = go2.sum(dim=(0, 2)) if ctx.has_bias else None
        return grad_x, grad_w, grad_b


def _is_plain_1x1(conv: nn.Conv2d) -> bool:
    return (
        conv.kernel_size == (1, 1)
        and conv.stride == (1, 1)
        and conv.padding == (0, 0)
        and conv.dilation == (1, 1)
        and conv.groups == 1
    )


_orig_forward = nn.Conv2d.forward
_patched = False


def _patched_forward(self, x):
    # channels_last inputs: take the native muDNN conv, NOT this GEMM. The native
    # NHWC 1x1 conv is ~2-2.8x faster than NCHW AND keeps the output channels_last,
    # so the channels_last chain survives through the backbone. The matmul+view
    # GEMM below forces NCHW output and would break the chain at every 1x1 conv.
    if (x.dim() == 4 and _is_plain_1x1(self)
            and not x.is_contiguous(memory_format=torch.channels_last)):
        # weight.contiguous(): matmul is layout-agnostic, but DDP compares the
        # grad's strides against the bucket-view strides. After
        # model.to(memory_format=channels_last) the weight gets NHWC strides
        # while our reshape-based grad is NCHW; DDP then warns "Grad strides
        # do not match bucket view strides" and silently copies. Forcing
        # contiguous here is essentially free for matmul.
        w = self.weight.contiguous() if not self.weight.is_contiguous() else self.weight
        return _Conv1x1MmFn.apply(x, w, self.bias)
    return _orig_forward(self, x)


def patch_1x1_conv() -> bool:
    """Install the patch. Idempotent. Returns True if newly installed."""
    global _patched
    if _patched:
        return False
    nn.Conv2d.forward = _patched_forward
    _patched = True
    return True


def unpatch_1x1_conv() -> bool:
    """Restore the original nn.Conv2d.forward. Returns True if anything changed."""
    global _patched
    if not _patched:
        return False
    nn.Conv2d.forward = _orig_forward
    _patched = False
    return True
