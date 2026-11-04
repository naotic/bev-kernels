"""NHWC split DCNv2 forward: muDNN offset conv + hand-written deformable gather
(ext.modulated_deformable_im2col_nhwc_cuda) + muBLAS main GEMM.

This is the "split" architecture (offset conv and main GEMM stay as optimized GEMM
calls; only the F4-F5 gather+modulate is the custom kernel). Forward only for now;
a differentiable autograd.Function + NHWC backward (col2im) is the next step.

Layout: x [B,H,W,Cin], w_off [3T,Cin,kH,kW], b_off [3T], weight [Cout,Cin,kH,kW].
Returns out [B,Ho,Wo,Cout].
"""
import os
import torch
import torch.nn.functional as F
import torch.nn.grad as tnng


def out_size(H, pad, dil, k, stride):
    return (H + 2 * pad - (dil * (k - 1) + 1)) // stride + 1


def gather_cols_nhwc(x_nhwc, offset, mask=None, kH=3, kW=3, stride=1, padding=1, dilation=1):
    """Deformable gather -> Col [M, T*Cin] (k = t*Cin+ci).

    Fused input: pass the raw offset-conv output y [B,Ho,Wo,3T] as `offset` (mask=None)
    -> the kernel reads offset from y[:,:,:,0:2T] and mask=sigmoid(y[...,2T:]) on the
    fly (the ext launcher detects the 3T last-dim). Non-fused: pass offset[2T]+mask[T].
    """
    import _modulated_deform_conv_ext as ext
    B, H, W, Cin = x_nhwc.shape
    Tt = kH * kW
    Ho = out_size(H, padding, dilation, kH, stride)
    Wo = out_size(W, padding, dilation, kW, stride)
    M = B * Ho * Wo
    off_c = offset.contiguous()
    mask_c = (mask if mask is not None else offset).contiguous()  # fused: dummy (=y), unused
    # empty (not zeros): the gather kernel writes EVERY element of col [M, T*Cin]
    # exactly once before the GEMM reads it (out-of-bounds taps write 0, they are
    # not skipped; grid covers all (b,oh,ow,t) and the vec+tail loops cover all
    # Cin channels). The memset is pure redundant work. Math is unchanged.
    col = torch.empty(M, Tt * Cin, device=x_nhwc.device, dtype=x_nhwc.dtype)
    ext.modulated_deformable_im2col_nhwc_cuda(
        x_nhwc.contiguous(), off_c, mask_c,
        B, Cin, H, W, Ho, Wo, kH, kW, padding, padding, stride, stride,
        dilation, dilation, 1, col)
    return col


def col2im_nhwc(dcol, x_nhwc, offset, mask=None, kH=3, kW=3, stride=1, padding=1, dilation=1):
    """Backward col2im. Returns (grad_x_samp, grad_offset, gmask_raw), all fp32.
    gmask_raw is the grad wrt the PRE-sigmoid mask (sigmoid backward folded in).

    Fused input: pass raw conv output y [B,Ho,Wo,3T] as `offset` (mask=None) — the
    perelem kernels read offset/mask from y (mask sigmoid'd inside). For non-perelem
    modes the separate offset/mask are materialized from y here.
    """
    import _modulated_deform_conv_ext as ext
    import os
    B, H, W, Cin = x_nhwc.shape
    Tt = kH * kW
    Ho = out_size(H, padding, dilation, kH, stride)
    Wo = out_size(W, padding, dilation, kW, stride)
    fast = (kH == 3 and kW == 3 and stride == 1 and dilation == 1 and Cin % 32 == 0)
    mode = os.environ.get("DCN_COL2IM_MODE")
    if mode is None:
        mode = "perelem_smem" if (dcol.dtype == torch.float16 and Cin <= 256) else "perelem"
    perelem_path = fast and mode in ("perelem", "perelem_smem")
    is_y = mask is None    # `offset` is the raw conv output y [B,Ho,Wo,3T]
    # non-perelem kernels need separate offset[2T] + POST-sigmoid mask[T]
    if is_y and not perelem_path:
        mask = torch.sigmoid(offset[..., 2 * Tt:])
        offset = offset[..., :2 * Tt]
        is_y = False
    dcol = dcol.contiguous(); xc = x_nhwc.contiguous()
    offc = offset.contiguous()
    mkc = (offset if is_y else mask).contiguous()  # fused: dummy (=y), unused by kernel
    gX = torch.zeros(B, H, W, Cin, device=x_nhwc.device, dtype=torch.float32)
    # gOff/gMask atomic-accumulated -> zero them (tiny tensors)
    gOff = torch.zeros(B, Ho, Wo, 2 * Tt, device=x_nhwc.device, dtype=torch.float32)
    gMask = torch.zeros(B, Ho, Wo, Tt, device=x_nhwc.device, dtype=torch.float32)
    # perelem (one thread per (m,t,ci)) is fastest (compute hides atomics); fp16 &
    # Cin<=256 -> perelem_smem (+6%). fused_smem/split/fused kept for comparison.
    if fast and mode == "perelem":
        # one thread per (m,t,ci); per-element geometry recompute hides the atomics
        ext.modulated_deformable_col2im_perelem_nhwc_cuda(
            dcol, xc, offc, mkc, B, Cin, H, W, Ho, Wo, kH, kW, padding, padding,
            stride, stride, dilation, dilation, 1, gX, gOff, gMask)
    elif fast and mode == "perelem_smem":
        # smem-shared geometry — no consistent win (fp16/256 +6%, fp32/512 -6~15%);
        # geometry redundancy is load-bearing (hides atomics). Kept for reference.
        ext.modulated_deformable_col2im_perelem_smem_nhwc_cuda(
            dcol, xc, offc, mkc, B, Cin, H, W, Ho, Wo, kH, kW, padding, padding,
            stride, stride, dilation, dilation, 1, gX, gOff, gMask)
    elif fast and mode == "fused_smem":
        # ONE kernel: grad_X smem-tiled + grad_off/mask (B4-B9 fully fused)
        ext.modulated_deformable_col2im_fused_smem_nhwc_cuda(
            dcol, xc, offc, mkc, B, Cin, H, W, Ho, Wo, kH, kW, padding, padding,
            stride, stride, dilation, dilation, 1, gX, gOff, gMask)
    elif fast and mode == "split":
        ext.modulated_deformable_col2im_coord_nhwc_cuda(
            dcol, xc, offc, mkc, B, Cin, H, W, Ho, Wo, kH, kW, padding, padding,
            stride, stride, dilation, dilation, 1, gOff, gMask)
        ext.modulated_deformable_gradx_smem_nhwc_cuda(
            dcol, offc, mkc, B, Cin, H, W, Ho, Wo, kH, kW, padding, padding,
            stride, stride, dilation, dilation, 1, gX)
    else:
        ext.modulated_deformable_col2im_nhwc_cuda(
            dcol, xc, offc, mkc, B, Cin, H, W, Ho, Wo, kH, kW, padding, padding,
            stride, stride, dilation, dilation, 1, gX, gOff, gMask)
    # perelem kernels FUSE the sigmoid backward (gMask already = grad wrt pre-sigmoid
    # mask). Other modes output grad wrt post-sigmoid mask -> apply it here.
    if not (fast and mode in ("perelem", "perelem_smem")):
        m_f = mkc.float()
        gMask = gMask * m_f * (1.0 - m_f)
    return gX, gOff, gMask  # gMask == grad wrt PRE-sigmoid mask (raw)


def dcn_split_forward_nhwc(x_nhwc, w_off, b_off, weight, bias=None,
                           stride=1, padding=1, dilation=1):
    import _modulated_deform_conv_ext as ext
    B, H, W, Cin = x_nhwc.shape
    Cout, _, kH, kW = weight.shape
    Tt = kH * kW
    Ho = out_size(H, padding, dilation, kH, stride)
    Wo = out_size(W, padding, dilation, kW, stride)
    M = B * Ho * Wo

    # --- F1-F3: offset conv via muDNN (NHWC in as a channels_last view) ---
    x_cl = x_nhwc.permute(0, 3, 1, 2)                      # [B,Cin,H,W] channels_last
    y = F.conv2d(x_cl, w_off, b_off, stride, padding, dilation)  # [B,3T,Ho,Wo]
    y = y.permute(0, 2, 3, 1)                              # [B,Ho,Wo,3T]
    offset = y[..., :2 * Tt].contiguous()
    mask = torch.sigmoid(y[..., 2 * Tt:]).contiguous()

    # --- F4-F5: deformable gather + modulation (custom kernel) -> Col[M, T*Cin] ---
    # empty (not zeros): the gather kernel fully writes col before the GEMM reads it
    # (see gather_cols_nhwc for the coverage rationale). Redundant memset removed.
    col = torch.empty(M, Tt * Cin, device=x_nhwc.device, dtype=x_nhwc.dtype)
    ext.modulated_deformable_im2col_nhwc_cuda(
        x_nhwc, offset, mask, B, Cin, H, W, Ho, Wo, kH, kW,
        padding, padding, stride, stride, dilation, dilation, 1, col)

    # --- F6: main GEMM (muBLAS) ---  w_flat[k, co], k = t*Cin+ci
    w_flat = weight.permute(2, 3, 1, 0).reshape(Tt * Cin, Cout).contiguous().to(x_nhwc.dtype)
    out = torch.mm(col, w_flat).view(B, Ho, Wo, Cout)
    if bias is not None:
        out = out + bias.view(1, 1, 1, -1)
    return out


class DCNSplitNHWCFunction(torch.autograd.Function):
    """Differentiable NHWC DCNv2: muDNN offset conv + custom gather + muBLAS GEMM
    forward; custom col2im + muDNN conv-backward + muBLAS GEMM backward.

    Inputs: x_nhwc [B,H,W,Cin], w_off [3T,Cin,kH,kW], b_off [3T], weight
    [Cout,Cin,kH,kW], bias [Cout] or None.  Output: out [B,Ho,Wo,Cout].
    """

    @staticmethod
    def forward(ctx, x_nhwc, w_off, b_off, weight, bias, stride, padding, dilation,
                bn_scale=None, bn_shift=None, act_relu=False):
        B, H, W, Cin = x_nhwc.shape
        Cout, _, kH, kW = weight.shape
        Tt = kH * kW
        Ho = out_size(H, padding, dilation, kH, stride)
        Wo = out_size(W, padding, dilation, kW, stride)
        M = B * Ho * Wo

        # y = raw offset-conv output [B,Ho,Wo,3T]; sigmoid(mask) + offset-read are
        # FUSED into the gather (no separate sigmoid/slice/contiguous). .contiguous()
        # is a no-op when the conv output is channels_last (the common fuse_c path).
        y = F.conv2d(x_nhwc.permute(0, 3, 1, 2), w_off, b_off, stride, padding, dilation)
        y = y.permute(0, 2, 3, 1).contiguous()

        col = gather_cols_nhwc(x_nhwc, y, None, kH, kW, stride, padding, dilation)
        dt = x_nhwc.dtype
        w_flat = weight.permute(2, 3, 1, 0).reshape(Tt * Cin, Cout).contiguous().to(dt)

        # --- F6: main GEMM (muBLAS), optionally with a fused FrozenBN(affine)+ReLU
        # epilogue. FrozenBN scale/shift are frozen per-OUTPUT-channel buffers, so
        #   relu(scale*(col@w_flat + bias) + shift)
        #     == relu(col@(w_flat*scale) + (shift + scale*bias))
        # which muBLAS runs as a single GEMM-BIAS-RELU epilogue (torch._addmm_
        # activation). Bit-exact to the unfused GEMM->FrozenBN->ReLU (tf32/fp noise
        # only). bn_scale=None -> plain GEMM (baseline, byte-identical).
        relu_mask = None
        if bn_scale is not None:
            scale_dt = bn_scale.to(dt)
            w_eff = (w_flat * scale_dt.view(1, -1)).contiguous()
            if bias is not None:
                bias_eff = (bn_shift + bias.to(bn_shift.dtype) * bn_scale).to(dt)
            else:
                bias_eff = bn_shift.to(dt)
            if act_relu:
                out2d = torch._addmm_activation(bias_eff, col, w_eff,
                                                beta=1, alpha=1, use_gelu=False)
                relu_mask = out2d > 0
            else:
                out2d = torch.addmm(bias_eff, col, w_eff)
            out = out2d.view(B, Ho, Wo, Cout)
        else:
            out = torch.mm(col, w_flat).view(B, Ho, Wo, Cout)
            if bias is not None:
                out = out + bias.view(1, 1, 1, -1)

        ctx.has_bn = bn_scale is not None
        ctx.act_relu = bool(act_relu) and ctx.has_bn

        # By default save Col so backward reuses it (no re-gather). Col is
        # [M,T*Cin] and, across all ResNet DCN layers, dominates the fuse_c memory
        # footprint (~+8GB in full BEVFormer -> OOM at samples_per_gpu=2). Set
        # DCN_RECOMPUTE_COLS=1 to NOT save Col and instead re-gather it in backward
        # from the saved x + y (trades a re-gather for ~that memory). Default ON so
        # fuse_c fits the full BEVFormer batch; =0 to save Col (faster bwd, more mem).
        recompute = os.environ.get("DCN_RECOMPUTE_COLS", "1").lower() in (
            "1", "true", "yes", "on")
        ctx.recompute_cols = recompute
        saved = [x_nhwc, y, w_off, weight]
        if not recompute:
            saved.append(col)
        # epilogue backward needs the frozen scale and the relu mask
        if ctx.has_bn:
            saved.append(bn_scale)
            if ctx.act_relu:
                saved.append(relu_mask)
        ctx.save_for_backward(*saved)
        ctx.stride, ctx.padding, ctx.dilation = stride, padding, dilation
        ctx.with_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, grad_out):
        st, pd, dl = ctx.stride, ctx.padding, ctx.dilation
        saved = list(ctx.saved_tensors)
        x, y, w_off, weight = saved[0], saved[1], saved[2], saved[3]
        Cout, _, kH, kW = weight.shape
        idx = 4
        if getattr(ctx, "recompute_cols", False):
            # re-gather Col from saved x + raw conv output y (saves Col memory)
            cols = gather_cols_nhwc(x, y, None, kH, kW, st, pd, dl)
        else:
            cols = saved[idx]; idx += 1  # cols saved from fwd
        bn_scale = relu_mask = None
        if getattr(ctx, "has_bn", False):
            bn_scale = saved[idx]; idx += 1
            if getattr(ctx, "act_relu", False):
                relu_mask = saved[idx]; idx += 1
        B, H, W, Cin = x.shape
        Tt = kH * kW
        Ho = out_size(H, pd, dl, kH, st)
        Wo = out_size(W, pd, dl, kW, st)
        M = B * Ho * Wo
        K = Tt * Cin
        dt = x.dtype

        # Fused-epilogue backward: out = relu?(scale*(col@w_flat + bias) + shift),
        # so grad wrt the raw GEMM output = grad_out * relu_mask * scale. Folding
        # both into dO makes the rest of the backward (dCol, grad_weight, grad_bias)
        # identical to the unfused path and bit-exact (the weight/bias grads pick up
        # the per-channel scale through dO). bn_scale=None -> dO == grad_out.
        go = grad_out.reshape(M, Cout)
        if getattr(ctx, "has_bn", False):
            if getattr(ctx, "act_relu", False):
                go = go * relu_mask.reshape(M, Cout)
            go = go * bn_scale.to(go.dtype).view(1, -1)
        dO = go.to(dt)
        w_flat = weight.permute(2, 3, 1, 0).reshape(K, Cout).contiguous().to(dt)  # [K,Cout]

        # --- F6 backward (reuse saved cols, no re-gather) ---
        dCol = torch.mm(dO, w_flat.t())                                          # [M,K]
        grad_weight = torch.mm(cols.t().float(), dO.float()) \
            .view(kH, kW, Cin, Cout).permute(3, 2, 0, 1).contiguous().to(dt)
        grad_bias = dO.float().sum(0).to(dt) if ctx.with_bias else None

        # --- F4-F5 backward (col2im); fused: reads offset/mask from y, sigmoid bwd folded in ---
        gX_samp, gOff, gmask_raw = col2im_nhwc(dCol, x, y, None, kH, kW, st, pd, dl)

        # --- F1 offset-conv backward (regular conv) ---
        dY = torch.cat([gOff, gmask_raw], dim=-1).permute(0, 3, 1, 2).contiguous()  # [B,3T,Ho,Wo]
        x_nchw = x.permute(0, 3, 1, 2).contiguous().float()
        w_off_f = w_off.float()
        stt, pdt, dlt = (st, st), (pd, pd), (dl, dl)
        grad_x_off = tnng.conv2d_input(x_nchw.shape, w_off_f, dY, stt, pdt, dlt)
        grad_w_off = tnng.conv2d_weight(x_nchw, w_off_f.shape, dY, stt, pdt, dlt)
        grad_b_off = dY.sum(dim=(0, 2, 3))
        grad_x_off = grad_x_off.permute(0, 2, 3, 1)                              # NHWC

        grad_x = (gX_samp + grad_x_off).to(dt)
        # inputs: x, w_off, b_off, weight, bias, stride, padding, dilation,
        #         bn_scale, bn_shift, act_relu  (last 6 are non-differentiable)
        return (grad_x, grad_w_off.to(dt), grad_b_off.to(dt), grad_weight,
                grad_bias, None, None, None, None, None, None)


def dcn_split_nhwc(x_nhwc, w_off, b_off, weight, bias=None, stride=1, padding=1, dilation=1,
                   bn_scale=None, bn_shift=None, act_relu=False):
    """Autograd-enabled NHWC DCNv2 forward (differentiable).

    Optional fused epilogue: pass bn_scale/bn_shift (FrozenBN per-out-channel
    affine) and act_relu=True to fold a following FrozenBN(+ReLU) into the main
    GEMM (muBLAS GEMM-BIAS-RELU). bn_scale=None -> plain GEMM (baseline)."""
    return DCNSplitNHWCFunction.apply(x_nhwc, w_off, b_off, weight, bias,
                                      stride, padding, dilation,
                                      bn_scale, bn_shift, act_relu)


def _as_int(v):
    return v[0] if isinstance(v, (tuple, list)) else v


def dcn_apply_nchw(x_nchw, w_off, b_off, weight, bias=None, stride=1, padding=1, dilation=1,
                   bn_scale=None, bn_shift=None, act_relu=False):
    """NCHW-in / NCHW-out adapter over the hand-CUDA NHWC DCNv2 (differentiable).

    Drop-in for the existing NCHW path: weights stay [Cout,Cin,kH,kW], conv_offset
    weight/bias passed through (offset conv computed inside). channels_last input
    -> permute is a free contiguous view (zero copy); output matches input format.
    Scope: groups==1, deform_groups==1, square/symmetric 3x3 stride/pad/dilation.

    Optional fused epilogue (bn_scale/bn_shift/act_relu): fold a following
    FrozenBN(+ReLU) into the main GEMM. See dcn_split_nhwc."""
    s, p, d = _as_int(stride), _as_int(padding), _as_int(dilation)
    is_cl = x_nchw.is_contiguous(memory_format=torch.channels_last)
    x_nhwc = x_nchw.permute(0, 2, 3, 1)
    if not is_cl:
        x_nhwc = x_nhwc.contiguous()
    out_nhwc = dcn_split_nhwc(x_nhwc, w_off, b_off, weight, bias, s, p, d,
                              bn_scale, bn_shift, act_relu)  # [B,Ho,Wo,Cout]
    out = out_nhwc.permute(0, 3, 1, 2)                                       # [B,Cout,Ho,Wo]
    return out if is_cl else out.contiguous()
